import argparse
import pathlib
import time
import torch
import json
import math
import imageio
import numpy as np
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Update this to match your loader
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score, confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from io import BytesIO
from dgl import load_graphs
from io import BytesIO

from datasets.fusiongallery import FusionGalleryDataset, FusionGalleryContrastive
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from datasets.solidletters_contrastive import SolidLettersContrastive
#from uvnet.models_SimCLR import Contrastive
from uvnet.models_DViCe import Contrastive
# from uvnet.models_MoCo import MoCo
# from uvnet.models_BYOL import BYOL
# from uvnet.models_DINO import DINOWithPrototypes
# from uvnet.models_Barlow_Twins import BarlowTwins
# from uvnet.models_VicReg import VICReg
# from uvnet.models_SimSiam import Contrastive
# from uvnet.models_Swav import Contrastive
from fvcore.nn import FlopCountAnalysis


#from occwl.io import load_shape_from_file  # corrected import
from util_visualize import retrieve_top_k, retrieve_and_visualize_queries, retrieve_single_sample 
from util_visualize import save_results, append_to_leaderboard, retrieve_top_k_cosine, retrieve_top_k_autoencoder 
from util_visualize import retrieve_top_k_faiss, retrieve_top_k_hnsw, retrieve_top_k_l2, retrieve_top_k_learned_matrix




def count_flops_forward_no_gnn(model, sample_batch, device="cuda"):
    """
    Approximate FLOPs per forward pass for model.model_1.
    Counts:
      - UVNetCurveEncoder
      - UVNetSurfaceEncoder
      - projection_layer
    Skips:
      - UVNetGraphEncoder (because it needs a DGL graph)
    """
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        g = sample_batch["graph"].to(device)
        g = model._permute_graph_data_channels(g)

        # Tensor inputs for encoders
        nfeat = g.ndata["x"][:, [0, 1, 2, 6], :, :]          # [N, 4, 10, 10]
        efeat = g.edata["x"][:, :model.model_1.crv_in_channels, :]          # [E, 3, 10]

        # Run once to get intermediate tensors
        crv_feat = model.model_1.curv_encoder(efeat)          # tensor
        srf_feat = model.model_1.surf_encoder(nfeat)          # tensor
        _, graph_emb = model.model_1.graph_encoder(g, srf_feat, crv_feat)  # tensor
        proj_out = model.model_1.projection_layer(graph_emb)  # tensor

        # FLOPs per module (tensors only)
        flops_curv = FlopCountAnalysis(model.model_1.curv_encoder, efeat).total()
        flops_surf = FlopCountAnalysis(model.model_1.surf_encoder, nfeat).total()
        flops_proj = FlopCountAnalysis(model.model_1.projection_layer, graph_emb).total()

    flops_total = flops_curv + flops_surf + flops_proj

    print(f"Curve encoder FLOPs:      {flops_curv:.2e}")
    print(f"Surface encoder FLOPs:    {flops_surf:.2e}")
    print(f"Projection head FLOPs:    {flops_proj:.2e}")
    print(f"Total (no GNN) FLOPs/fw:  {flops_total:.2e}")

    return flops_total


def compute_training_and_total_flops(flops_per_forward, 
                                     train_samples,
                                     val_samples,
                                     batch_size,
                                     epochs):
    # steps per epoch (ceil to include last smaller batch)
    train_steps_per_epoch = math.ceil(train_samples / batch_size)  # 31285/256 → 123
    val_steps_per_epoch   = math.ceil(val_samples   / batch_size)  # 7822/256  → 31

    train_steps = train_steps_per_epoch * epochs
    val_steps   = val_steps_per_epoch   * epochs

    # One training step ≈ forward + backward ≈ 2 × forward FLOPs
    flops_per_train_step = 2 * flops_per_forward

    training_flops   = flops_per_train_step * train_steps
    validation_flops = flops_per_forward * val_steps
    total_flops      = training_flops + validation_flops

    print("---------------------------------------------------")
    print(f"Train steps per epoch:       {train_steps_per_epoch}")
    print(f"Val steps   per epoch:       {val_steps_per_epoch}")
    print(f"Total train steps:           {train_steps}")
    print(f"Total val   steps:           {val_steps}")
    print("---------------------------------------------------")
    print(f"FLOPs per forward pass:      {flops_per_forward:.2e}")
    print(f"FLOPs per training step:     {flops_per_train_step:.2e}")
    print(f"Total TRAINING FLOPs:        {training_flops:.2e}")
    print(f"Total VALIDATION FLOPs:      {validation_flops:.2e}")
    print(f"Total (train + val) FLOPs:   {total_flops:.2e}")
    print("---------------------------------------------------")

    return {"flops_per_forward": flops_per_forward,
            "flops_per_train_step": flops_per_train_step,
            "training_flops": training_flops,
            "validation_flops": validation_flops,
            "total_flops": total_flops}



parser = argparse.ArgumentParser("UV-Net self-supervision with contrastive learning")
parser.add_argument("--traintest", choices=("train", "test"), default="train", help="Whether to train or test")
parser.add_argument("--testoption", choices=("full_test", "selective_test"), default="selective_test", help="Whether test on whole test set or selective test set")
parser.add_argument("--dataset", choices=("solidletters"), default="solidletters", help="Dataset to train on")
parser.add_argument("--dataset_path", type=str, default="/home/ar/CAD/D_SolidLetters/graph_with_eattr", help="Path to dataset") #"/home/ar/CAD/D_SolidLetters/graph_with_eattr"
parser.add_argument("--size_percentage", type=float, default=1.0, help="Percentage of data to load")
parser.add_argument("--temperature", type=float, default=0.1, help="Temperature to use in NTXentLoss")
parser.add_argument("--latent_dim", type=int, default=128, help="Latent dimension for UV-Net's embeddings")
parser.add_argument("--out_dim", type=int, default=64, help="Output dimension for SimCLR projection head")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
parser.add_argument("--num_workers", type=int, default=16, help="Number of workers for the dataloader")
parser.add_argument("--checkpoint", type=str, default="/home/ar/CAD/UV-Net/results/contrastive_retrieve_UpperCase/241025/110254/best.ckpt", help="Checkpoint file to load weights from for testing")
parser.add_argument("--experiment_name", type=str, default="contrastive_retrieve_UpperCase", help="Experiment name")
parser.add_argument("--eval_method", 
                    choices=("knn", "cosine", "learned", "euclidean", "hnsw", "faiss"), 
                    default=["knn", "cosine", "learned", "euclidean", "hnsw", "faiss"], 
                    help="Experiment name")
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

results_path = pathlib.Path(__file__).parent.joinpath("results", args.experiment_name)
results_path.mkdir(parents=True, exist_ok=True)
month_day = time.strftime("%d%m%y")
hour_min_second = time.strftime("%H%M%S")

checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=str(results_path.joinpath(month_day, hour_min_second)), filename="best", save_last=True)
trainer = Trainer.from_argparse_args(args, callbacks=[checkpoint_callback], logger=TensorBoardLogger(str(results_path), name=month_day, version=hour_min_second), accelerator="ddp", gpus=-1)

if args.dataset == "solidletters":
    Dataset = SolidLettersContrastive
else:
    raise ValueError("Unsupported dataset")


def test_augmented_dataset(dataset):
    loader = dataset.get_dataloader(batch_size=4, shuffle=False)
    batch = next(iter(loader))
    g1 = batch["graph"]
    g2 = batch["graph2"]
    print("Graph1 batch:", g1)
    print("Graph2 batch:", g2)
    print("Labels:", batch["label"])
    print("Filenames:", batch["filename"])
    assert g1.ndata["x"].shape == g2.ndata["x"].shape
    print("✓ Passed basic shape checks")

if args.traintest == "train":
    seed_everything(workers=True)
    print(f"""
          -----------------------------------------------------------------------------------
          D-ViCe Contrastive Learning
          -----------------------------------------------------------------------------------
          Logs written to results/{args.experiment_name}/{month_day}/{hour_min_second}
          To monitor logs, run:
          tensorboard --logdir results/{args.experiment_name}/{month_day}/{hour_min_second}
          -----------------------------------------------------------------------------------
          """)
    model = Contrastive(latent_dim=args.latent_dim, out_dim=args.out_dim)
    # model = DINOWithPrototypes(latent_dim=128, 
    #                            out_dim=128, 
    #                            num_prototypes=256, 
    #                            base_momentum=0.996, 
    #                            center_momentum=0.9, 
    #                            student_temp_init=0.1,
    #                            teacher_temp_init=0.07)                           

    #model = MoCo(latent_dim=args.latent_dim, out_dim=args.out_dim, temperature=args.temperature)
    # model = BYOL(latent_dim=args.latent_dim, out_dim=args.out_dim)
    # model = BarlowTwins(latent_dim=args.latent_dim, out_dim=args.out_dim)
    # model = VICReg(latent_dim=args.latent_dim, out_dim=args.out_dim)

    # === visualize LR schedule before training ===
    base_lr = 0.2
    batch_size = args.batch_size
    lr = (batch_size / 256) * base_lr
    warmup_epochs = 10
    max_epochs = getattr(args, "max_epochs", None) or 100
    final_lr_scale = 0.002 / lr

    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return current_epoch / float(warmup_epochs)
        progress = (current_epoch - warmup_epochs) / float(max_epochs - warmup_epochs)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return final_lr_scale + (1 - final_lr_scale) * cosine_decay

    # plot schedule
    lrs = [lr * lr_lambda(e) for e in range(max_epochs)]
    plt.figure(figsize=(6,4))
    plt.plot(range(max_epochs), lrs, label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title("Warmup + Cosine Decay LR Schedule")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # === end LR plot ===

    train_data = Dataset(root_dir=args.dataset_path, split="train", size_percentage=args.size_percentage)
    #test_augmented_dataset(train_data)
    val_data = Dataset(root_dir=args.dataset_path, split="val", size_percentage=args.size_percentage)
    train_loader = train_data.get_dataloader(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = val_data.get_dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = Contrastive(latent_dim=128, out_dim=64).cuda()
    dummy_proj = torch.randn(256, 128).cuda()
    flops_proj = FlopCountAnalysis(model.model_1.projection_layer, dummy_proj).total()
    print("Projection FLOPs:", flops_proj/1e6, "MFLOPs")
    input()

    # num_params = sum(p.numel() for p in model.parameters())
    # print(f"Model has {num_params/1e6:.2f} million parameters")
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Model has {trainable_params/1e6:.2f} million trainable parameters")
    # input()

    # get one batch
    # after you create model and train_loader
    model = Contrastive(latent_dim=args.latent_dim, out_dim=args.out_dim)

    ##Count FLOPs
    # sample_batch = next(iter(train_loader))
    # flops_per_forward = count_flops_forward_no_gnn(model, sample_batch)
    # print("Using FLOPs per forward (no GNN):", f"{flops_per_forward:.2e}")

    # train_samples = len(train_loader.dataset)
    # batch_size = args.batch_size
    # epochs = args.max_epochs  # or what you use

    # stats = compute_training_and_total_flops(flops_per_forward=flops_per_forward,   # from count_flops_forward_no_gnn
    #                                          train_samples=31285,
    #                                          val_samples=7822,
    #                                          batch_size=256,
    #                                          epochs=1000)
    # print(stats)



    trainer.fit(model, train_loader, val_loader)

else:
    assert args.checkpoint != "", "Expected --checkpoint argument"
    model = Contrastive.load_from_checkpoint(args.checkpoint)
    # model = DINOWithPrototypes(latent_dim=128, 
    #                            out_dim=128, 
    #                            num_prototypes=256, 
    #                            base_momentum=0.996, 
    #                            center_momentum=0.9, 
    #                            student_temp_init=0.1,
    #                            teacher_temp_init=0.07) 
    # model = MoCo.load_from_checkpoint(args.checkpoint)
    # model = BYOL.load_from_checkpoint(args.checkpoint)
    # model = BarlowTwins(latent_dim=args.latent_dim, out_dim=args.out_dim)
    # model = VICReg(latent_dim=args.latent_dim, out_dim=args.out_dim)
    model = model.cuda()

    # train_data = Dataset(root_dir=args.dataset_path, split="train", shape_type="upper")
    train_data = Dataset(root_dir=args.dataset_path, split="train", size_percentage=args.size_percentage)
    train_loader = train_data.get_dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    train_outputs = model.get_embeddings_from_dataloader(train_loader)

    if args.testoption=="full_test":
        test_data = Dataset(root_dir=args.dataset_path, split="test", size_percentage=args.size_percentage)
        test_loader = test_data.get_dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
        test_outputs = model.get_embeddings_from_dataloader(test_loader)

        if "knn" in args.eval_method:
            json_path = str(results_path / f"{month_day}_{hour_min_second}_results_knn.json")
            csv_path = str(results_path / f"{month_day}_{hour_min_second}_results_knn.csv")
            leaderboard_path = str(results_path / f"{month_day}_{hour_min_second}_leaderboard_knn.csv")
            retrieval_results_knn, metrics_knn, conf_mat_knn = retrieve_top_k(test_outputs, train_outputs, top_k=10)
            save_results(retrieval_results_knn, metrics_knn, conf_mat_knn, json_path=json_path, csv_path=csv_path)
            append_to_leaderboard(metrics_knn, args.experiment_name, leaderboard_csv=leaderboard_path)

        if "cosine" in args.eval_method:
            json_path = str(results_path / f"{month_day}_{hour_min_second}_results_cosine.json")
            csv_path = str(results_path / f"{month_day}_{hour_min_second}_results_cosine.csv")
            leaderboard_path = str(results_path / f"{month_day}_{hour_min_second}_leaderboard_cosine.csv")
            retrieval_results_cosine, metrics_cosine, conf_mat_cosine = retrieve_top_k_cosine(test_outputs, train_outputs, top_k=10)
            save_results(retrieval_results_cosine, metrics_cosine, conf_mat_cosine, json_path=json_path, csv_path=csv_path)
            append_to_leaderboard(metrics_cosine, args.experiment_name, leaderboard_csv=leaderboard_path)
        
        if "learned" in args.eval_method:
            json_path = str(results_path / f"{month_day}_{hour_min_second}_results_learned.json")
            csv_path = str(results_path / f"{month_day}_{hour_min_second}_results_learned.csv")
            leaderboard_path = str(results_path / f"{month_day}_{hour_min_second}_leaderboard_learned.csv")
            retrieval_results_learn, metrics_learn, conf_mat_learn = retrieve_top_k_learned_matrix(test_outputs, train_outputs, top_k=10)
            save_results(retrieval_results_learn, metrics_learn, conf_mat_learn, json_path=json_path, csv_path=csv_path)
            append_to_leaderboard(metrics_learn, args.experiment_name, leaderboard_csv=leaderboard_path)

        if "euclidean" in args.eval_method:
            json_path = str(results_path / f"{month_day}_{hour_min_second}_results_euclidean.json")
            csv_path = str(results_path / f"{month_day}_{hour_min_second}_results_euclidean.csv")
            leaderboard_path = str(results_path / f"{month_day}_{hour_min_second}_leaderboard_euclidean.csv")
            retrieval_results_euclidean, metrics_euclidean, conf_mat_euclidean = retrieve_top_k_l2(test_outputs, train_outputs, top_k=10)
            save_results(retrieval_results_euclidean, metrics_euclidean, conf_mat_euclidean, json_path=json_path, csv_path=csv_path)
            append_to_leaderboard(metrics_euclidean, args.experiment_name, leaderboard_csv=leaderboard_path)

        if "hnsw" in args.eval_method:
            json_path = str(results_path / f"{month_day}_{hour_min_second}_results_hnsw.json")
            csv_path = str(results_path / f"{month_day}_{hour_min_second}_results_hnsw.csv")
            leaderboard_path = str(results_path / f"{month_day}_{hour_min_second}_leaderboard_hnsw.csv")
            retrieval_results_hnsw, metrics_hnsw, conf_mat_hnsw = retrieve_top_k_hnsw(test_outputs, train_outputs, top_k=10)
            save_results(retrieval_results_hnsw, metrics_hnsw, conf_mat_hnsw, json_path=json_path, csv_path=csv_path)
            append_to_leaderboard(metrics_hnsw, args.experiment_name, leaderboard_csv=leaderboard_path)
        
        if "faiss" in args.eval_method:
            json_path = str(results_path / f"{month_day}_{hour_min_second}_results_faiss.json")
            csv_path = str(results_path / f"{month_day}_{hour_min_second}_results_faiss.csv")
            leaderboard_path = str(results_path / f"{month_day}_{hour_min_second}_leaderboard_faiss.csv")
            retrieval_results_faiss, metrics_faiss, conf_mat_faiss = retrieve_top_k_faiss(test_outputs, train_outputs, top_k=10)
            save_results(retrieval_results_faiss, metrics_faiss, conf_mat_faiss, json_path=json_path, csv_path=csv_path)
            append_to_leaderboard(metrics_faiss, args.experiment_name, leaderboard_csv=leaderboard_path)

        if "autoencoder" in args.eval_method:
            json_path = str(results_path / f"{month_day}_{hour_min_second}_results_autoencoder.json")
            csv_path = str(results_path / f"{month_day}_{hour_min_second}_results_autoencoder.csv")
            leaderboard_path = str(results_path / f"{month_day}_{hour_min_second}_leaderboard_autoencoder.csv")
            retrieval_results_autoencoder, metrics_autoencoder, conf_mat_autoencoder = retrieve_top_k_autoencoder(test_outputs, train_outputs, top_k=10)
            save_results(retrieval_results_autoencoder, metrics_autoencoder, conf_mat_autoencoder, json_path=json_path, csv_path=csv_path)
            append_to_leaderboard(metrics_autoencoder, args.experiment_name, leaderboard_csv=leaderboard_path)

    if args.testoption=="selective_test":
        # Load test.txt for multiple queries
        #test_txt_path = pathlib.Path(args.dataset_path) / "test_selective.txt"
        test_selective_data = Dataset(root_dir=args.dataset_path, split="selective", shape_type="upper")
        test_selective_loader = test_selective_data.get_dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
        test_selective_outputs = model.get_embeddings_from_dataloader(test_selective_loader)
        retrieve_and_visualize_queries(test_selective_outputs, 
                                       train_outputs, 
                                       args.dataset_path,
                                       output_csv=str(results_path / f"{month_day}_{hour_min_second}_top10.csv"), 
                                       top_k=10)
        
        # retrieve_single_sample(query_outputs=test_selective_outputs, train_outputs=train_outputs, top_k=10)



