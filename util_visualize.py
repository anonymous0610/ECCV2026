import os
import numpy as np
import pathlib
import time
import torch
import torch.nn.functional as F
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns  # Update this to match your loader
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
import imageio
from dgl import load_graphs
from io import BytesIO
import imageio
#from occwl.io import load_shape_from_file  # corrected import
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, roc_auc_score
from PIL import Image
from sklearn.metrics.pairwise import euclidean_distances
import hnswlib
import faiss
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.random_projection import SparseRandomProjection
from sklearn.utils import murmurhash3_32
from sklearn.preprocessing import binarize
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# # ------------------------- Retrieval & Evaluation Functions -------------------------

def normalize_embeddings(embeddings):
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.tensor(embeddings, dtype=torch.float32).cuda()
    return F.normalize(embeddings, dim=1)

def compute_similarity(query_embeddings, db_embeddings):
    query_embeddings = normalize_embeddings(query_embeddings)
    db_embeddings = normalize_embeddings(db_embeddings)
    return torch.matmul(query_embeddings, db_embeddings.T)

def retrieve_top_k(query_outputs, db_outputs, top_k=5):
    start_time = time.time()
    sims = compute_similarity(query_outputs['embeddings'], db_outputs['embeddings'])
    topk_vals, topk_inds = torch.topk(sims, k=top_k, dim=1)

    retrieval_results = []
    correct_at_1 = 0
    correct_at_5 = 0
    true_positives = 0
    y_true = []
    y_pred = []
    y_scores = []

    for i, indices in enumerate(topk_inds):
        query_label = int(query_outputs['labels'][i])
        query_name = query_outputs['filenames'][i]
        matched_files = []
        match_flags = []

        top_labels = [int(db_outputs['labels'][j]) for j in indices]
        top_files = [db_outputs['filenames'][j] for j in indices]
        scores = topk_vals[i].tolist()

        match_at_1 = (top_labels[0] == query_label)
        match_at_5 = query_label in top_labels

        if match_at_1:
            correct_at_1 += 1
        if match_at_5:
            correct_at_5 += 1

        true_positives += sum([lbl == query_label for lbl in top_labels])
        y_true.append(query_label)
        y_pred.append(top_labels[0])
        y_scores.extend(scores)

        for j, f in enumerate(top_files):
            matched_files.append(f)
            match_flags.append(int(top_labels[j] == query_label))

        retrieval_results.append({
            "query": query_name,
            "query_label": query_label,
            "top_k_files": matched_files,
            "top_k_labels": top_labels,
            "match_flags": match_flags,
            "scores": scores
        })

    total = len(query_outputs['labels'])
    recall_at_1 = correct_at_1 / total
    recall_at_5 = correct_at_5 / total
    top1_matches = [1 if y == p else 0 for y, p in zip(y_true, y_pred)]
    top1_scores = [s[0] for s in topk_vals.tolist()]  # first score from each row
    auc = roc_auc_score(top1_matches, top1_scores) if len(set(top1_matches)) > 1 else None



    conf_mat = confusion_matrix(y_true, y_pred)
    elapsed = time.time() - start_time

    metrics = {"total_queries": total, 
               "true_positives": true_positives, 
               "recall@1": recall_at_1, 
               "recall@5": recall_at_5,
               "auc_roc": auc,
               "retrieval_time_sec": elapsed}

    return retrieval_results, metrics, conf_mat

def convert_to_python_types(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def save_results(results, metrics, conf_mat, json_path="retrieval_results.json", csv_path="retrieval_results.csv"):
    safe_results = [{k: convert_to_python_types(v) for k, v in item.items()} for item in results]
    safe_metrics = {k: convert_to_python_types(v) for k, v in metrics.items()}

    with open(json_path, "w") as f:
        json.dump({"results": safe_results, "metrics": safe_metrics}, f, indent=2)

    rows = []
    for item in safe_results:
        row = {"query": item["query"], 
               "query_label": item["query_label"], 
               "top_k_files": "|".join(item["top_k_files"]), 
               "top_k_labels": "|".join(map(str, item["top_k_labels"])), 
               "match_flags": "|".join(map(str, item["match_flags"])), 
               "scores": "|".join(f"{s:.4f}" for s in item["scores"])}
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=False, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix.png")
    print("✅ Saved confusion_matrix.png")

    print(f"✅ Results saved to: {json_path}, {csv_path}")
    print(f"📊 Metrics: {safe_metrics}")

# ------------------------- Leaderboard Logging -------------------------

def append_to_leaderboard(metrics, experiment_name, leaderboard_csv="leaderboard.csv"):
    row = {"experiment_name": experiment_name, **metrics}
    row = {k: convert_to_python_types(v) for k, v in row.items()}
    file_exists = os.path.exists(leaderboard_csv)
    df = pd.DataFrame([row])
    df.to_csv(leaderboard_csv, mode='a', header=not file_exists, index=False)
    print(f"📋 Appended to leaderboard: {leaderboard_csv}")


# ------------------------- Batch Query Retrieval Script with CSV + Visualization -------------------------
# def plot_graph_from_bin(ax, bin_path, title="", elev=20, azim=45):
#     try:
#         graph = load_graphs(str(bin_path))[0][0]
#         node_feats = graph.ndata["x"]
#         node_pos = node_feats[..., :3].mean(dim=(1, 2)).cpu().numpy()
#         x, y, z = node_pos[:, 0], node_pos[:, 1], node_pos[:, 2]
#         src, dst = graph.edges()
#         src, dst = src.numpy(), dst.numpy()

#         ax.scatter(x, y, z, c='blue', s=5)
#         for s, d in zip(src, dst):
#             ax.plot([x[s], x[d]], [y[s], y[d]], [z[s], z[d]], color='gray', linewidth=0.5)

#         ax.view_init(elev=elev, azim=azim)
#         max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
#         mid = lambda v: (v.max()+v.min()) * 0.5
#         ax.set_xlim(mid(x) - max_range, mid(x) + max_range)
#         ax.set_ylim(mid(y) - max_range, mid(y) + max_range)
#         ax.set_zlim(mid(z) - max_range, mid(z) + max_range)

#         ax.set_title(title, fontsize=8)
#         ax.axis("off")
#     except Exception as e:
#         print(f"⚠️ Failed to render graph at {bin_path}: {e}")
#         ax.text(0.5, 0.5, 0.5, "(Empty)", color="red", ha='center')
#         ax.set_title(f"{title} (Empty)", fontsize=8)
#         ax.axis("off")
'''
def plot_cad_from_bin(ax, bin_path, title="", elev=20, azim=45):
    try:
        shape = ShapeLoader.load_shape_from_file(str(bin_path))  # updated usage
        viewer = Axes3DViewer(ax)
        viewer.set_camera(elev=elev, azim=azim)
        viewer.display_shape(shape)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    except Exception as e:
        print(f"⚠️ Failed to render CAD shape at {bin_path}: {e}")
        ax.text(0.5, 0.5, 0.5, "(Empty)", color="red", ha='center')
        ax.set_title(f"{title} (Empty)", fontsize=8)
        ax.axis("off")
'''
# def retrieve_and_visualize_queries(test_outputs, train_outputs, dataset_path, output_csv="top10_retrievals.csv", top_k=10):
#     records = []
#     fig_dir = pathlib.Path("retrieval_visuals")
#     fig_dir.mkdir(exist_ok=True)

#     for i, query_embedding in enumerate(test_outputs['embeddings']):
#         query_name = str(test_outputs['filenames'][i])
#         query_tensor = normalize_embeddings(torch.tensor(query_embedding).unsqueeze(0).cuda())

#         try:
#             db_embeddings = normalize_embeddings(train_outputs['embeddings'])
#             sims = torch.matmul(query_tensor, db_embeddings.T)[0]
#             topk_vals, topk_inds = torch.topk(sims, k=top_k)
#             top_matches = [str(train_outputs['filenames'][idx]) for idx in topk_inds.tolist()]
#             scores = [topk_vals[i].item() for i in range(top_k)]

#             records.append({
#                 "query": query_name,
#                 "top_k_matches": "|".join(top_matches),
#                 "scores": "|".join(f"{s:.4f}" for s in scores)
#             })

#             fig = plt.figure(figsize=(3 * (top_k + 1), 3))
#             gs = gridspec.GridSpec(1, top_k + 1, figure=fig)

#             ax_query = fig.add_subplot(gs[0], projection='3d')
#             query_path = pathlib.Path(dataset_path) / f"{query_name}.bin"
#             #plot_cad_from_bin(ax_query, query_path, title="Query")
#             plot_graph_from_bin(ax_query, query_path, title="Query")

#             for j, match_name in enumerate(top_matches):
#                 ax_match = fig.add_subplot(gs[j + 1], projection='3d')
#                 match_path = pathlib.Path(dataset_path) / f"{match_name}.bin"
#                 #plot_cad_from_bin(ax_match, match_path, title=f"{j+1}\n{scores[j]:.2f}")
#                 plot_graph_from_bin(ax_match, match_path, title=f"{j+1}\n{scores[j]:.2f}")

#             fig_path = fig_dir / f"{query_name}_retrieval.png"
#             plt.tight_layout()
#             plt.savefig(fig_path)
#             plt.close(fig)

#         except Exception as e:
#             print(f"❌ Retrieval or visualization error for query '{query_name}': {e}")

#     df = pd.DataFrame(records)
#     df.to_csv(output_csv, index=False)
#     print(f"✅ Saved top-10 retrievals to {output_csv} and visuals to {fig_dir}/")



def plot_graph_from_bin(ax, bin_path, title="", elev=20, azim=45):
    try:
        graph = load_graphs(str(bin_path))[0][0]
        node_feats = graph.ndata["x"]

        # -------------------------------
        # Handle both 2D and 3D UV-Net features
        # -------------------------------
        if node_feats.ndim == 4:
            # Surface patches: [N, nu, nv, C]
            node_pos = node_feats[..., :3].mean(dim=(1, 2)).cpu().numpy()
        elif node_feats.ndim == 2:
            # Curve/graph nodes: [N, C]
            node_pos = node_feats[:, :3].cpu().numpy()
        else:
            raise ValueError(f"Unsupported node feature shape: {node_feats.shape}")

        x, y, z = node_pos[:, 0], node_pos[:, 1], node_pos[:, 2]
        src, dst = graph.edges()
        src, dst = src.numpy(), dst.numpy()

        # -------------------------------
        # Plot nodes
        # -------------------------------
        ax.scatter(x, y, z, c='blue', s=4)

        # -------------------------------
        # MUCH FASTER: Edge plotting using Line3DCollection
        # -------------------------------
        segments = np.stack([
            np.column_stack([x[src], x[dst]]),
            np.column_stack([y[src], y[dst]]),
            np.column_stack([z[src], z[dst]])
        ], axis=-1)

        line_collection = Line3DCollection(segments, colors="gray", linewidths=0.3)
        ax.add_collection3d(line_collection)

        # -------------------------------
        # Normalize axis for consistent 3D scaling
        # -------------------------------
        max_range = np.ptp(node_pos, axis=0).max() / 2.0
        center = node_pos.mean(axis=0)
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=7)
        ax.set_axis_off()

    except Exception as e:
        print(f"⚠️ Failed to render graph at {bin_path}: {e}")
        ax.text(0.5, 0.5, 0.5, "(Empty)", color="red", ha='center')
        ax.set_title(f"{title} (Empty)", fontsize=7)
        ax.set_axis_off()


# def retrieve_and_visualize_queries(
#         test_outputs,
#         train_outputs,
#         dataset_path,
#         output_csv="top10_retrievals.csv",
#         top_k=10):

#     fig_dir = pathlib.Path("retrieval_visuals")
#     fig_dir.mkdir(exist_ok=True)

#     # -----------------------------------------------------------
#     # PRECOMPUTE & NORMALIZE THE DATABASE EMBEDDINGS ONCE (GPU)
#     # -----------------------------------------------------------
#     db_embeddings = torch.tensor(train_outputs['embeddings'], dtype=torch.float32).cuda()
#     db_embeddings = normalize_embeddings(db_embeddings)   # [N_train, D]

#     train_filenames = [str(f) for f in train_outputs['filenames']]

#     records = []

#     # Camera view for consistency
#     VIEW_ELEV = 20
#     VIEW_AZIM = 45

#     # -----------------------------------------------------------
#     # LOOP: PROCESS EACH QUERY
#     # -----------------------------------------------------------
#     for i, query_emb_np in enumerate(test_outputs['embeddings']):
#         query_name = str(test_outputs['filenames'][i])

#         try:
#             # Convert query embedding → GPU
#             query_tensor = torch.tensor(query_emb_np, dtype=torch.float32).unsqueeze(0).cuda()
#             query_tensor = normalize_embeddings(query_tensor)  # [1, D]

#             # -----------------------------
#             # Cosine similarity retrieval
#             # -----------------------------
#             sims = torch.nn.functional.cosine_similarity(query_tensor, db_embeddings, dim=1)
#             topk_vals, topk_inds = torch.topk(sims, k=top_k)

#             top_matches = [train_filenames[idx] for idx in topk_inds.tolist()]
#             scores = topk_vals.detach().cpu().numpy().tolist()

#             # --------------------------------
#             # Save to CSV (clean structured)
#             # --------------------------------
#             row = {"query": query_name}
#             for rank, (m, s) in enumerate(zip(top_matches, scores), 1):
#                 row[f"match_{rank}"] = m
#                 row[f"score_{rank}"] = float(s)
#             records.append(row)

#             # --------------------------------
#             # Visualization (2-row layout)
#             # --------------------------------
#             fig = plt.figure(figsize=(3 * top_k, 6))
#             gs = gridspec.GridSpec(2, top_k + 1, height_ratios=[1, 1], figure=fig)

#             # ---- Query (top-left cell) ----
#             ax_q = fig.add_subplot(gs[0, :], projection="3d")
#             query_path = pathlib.Path(dataset_path) / f"{query_name}.bin"

#             if not query_path.exists():
#                 print(f"⚠ Missing file: {query_path}")
#                 continue

#             plot_graph_from_bin(ax_q, query_path, title=f"Query: {query_name}")
#             ax_q.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
#             ax_q.set_axis_off()

#             # ---- Top-K matches (second row) ----
#             for j, (match_name, score) in enumerate(zip(top_matches, scores)):
#                 ax_m = fig.add_subplot(gs[1, j], projection='3d')
#                 match_path = pathlib.Path(dataset_path) / f"{match_name}.bin"

#                 if not match_path.exists():
#                     print(f"⚠ Missing match file: {match_path}")
#                     continue

#                 plot_graph_from_bin(ax_m, match_path, title=f"{j+1}: {score:.2f}")
#                 ax_m.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
#                 ax_m.set_axis_off()

#             # -----------------------------------
#             # Tight layout (CVPR-ready)
#             # -----------------------------------
#             plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)

#             fig_path = fig_dir / f"{query_name}_retrieval.png"
#             plt.savefig(fig_path, dpi=300, bbox_inches="tight")
#             plt.close(fig)

#         except Exception as e:
#             print(f"❌ Retrieval or visualization error for query '{query_name}': {e}")
#             continue

#     # -----------------------------------------------------------
#     # Write CSV
#     # -----------------------------------------------------------
#     df = pd.DataFrame(records)
#     df.to_csv(output_csv, index=False)
#     print(f"✅ Saved retrieval CSV to {output_csv}")
#     print(f"✅ Saved figures to: {fig_dir}/")

def retrieve_and_visualize_queries(
        test_outputs,
        train_outputs,
        dataset_path,
        output_csv="top7_retrievals.csv",
        top_k=7):

    fig_dir = pathlib.Path("retrieval_visuals")
    fig_dir.mkdir(exist_ok=True)

    # --- Precompute DB embeddings once ---
    db_embeddings = torch.tensor(train_outputs['embeddings'], dtype=torch.float32).cuda()
    db_embeddings = normalize_embeddings(db_embeddings)

    train_filenames = [str(f) for f in train_outputs['filenames']]
    records = []

    VIEW_ELEV = 20
    VIEW_AZIM = 45

    for i, query_emb_np in enumerate(test_outputs['embeddings']):
        query_name = str(test_outputs['filenames'][i])

        try:
            query_tensor = torch.tensor(query_emb_np, dtype=torch.float32).unsqueeze(0).cuda()
            query_tensor = normalize_embeddings(query_tensor)

            # --- similarity ---
            sims = torch.nn.functional.cosine_similarity(query_tensor, db_embeddings, dim=1)
            topk_vals, topk_inds = torch.topk(sims, k=top_k)

            top_matches = [train_filenames[idx] for idx in topk_inds.tolist()]
            scores = topk_vals.detach().cpu().numpy().tolist()

            # --- CSV output ---
            row = {"query": query_name}
            for rank, (m, s) in enumerate(zip(top_matches, scores), 1):
                row[f"match_{rank}"] = m
                row[f"score_{rank}"] = float(s)
            records.append(row)

            # ==========================================================
            # SINGLE ROW VISUALIZATION (Query + 7 retrieved models)
            # ==========================================================
            total_cols = top_k + 1   # 1 query + 7 retrieved
            fig = plt.figure(figsize=(3 * total_cols, 4))
            gs = gridspec.GridSpec(1, total_cols, figure=fig)

            # -------- Query ---------
            ax_q = fig.add_subplot(gs[0, 0], projection="3d")
            query_path = pathlib.Path(dataset_path) / f"{query_name}.bin"

            plot_graph_from_bin(ax_q, query_path, title=f"Query:\n{query_name}")
            ax_q.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
            ax_q.set_axis_off()

            # -------- Retrieved Results (Columns 2–8) ---------
            for j, (match_name, score) in enumerate(zip(top_matches, scores)):
                ax_m = fig.add_subplot(gs[0, j+1], projection='3d')
                match_path = pathlib.Path(dataset_path) / f"{match_name}.bin"

                plot_graph_from_bin(ax_m, match_path, 
                                    title=f"{j+1}: {match_name}\n({score:.2f})")
                ax_m.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
                ax_m.set_axis_off()

            plt.tight_layout(pad=0.2, w_pad=0.2)
            fig_path = fig_dir / f"{query_name}_retrieval_row.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        except Exception as e:
            print(f"❌ Retrieval or visualization error for '{query_name}': {e}")
            continue

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved CSV → {output_csv}")
    print(f"✅ Saved figures → {fig_dir}/")





'''
def retrieve_and_visualize_queries(test_outputs, train_outputs, dataset_path, output_csv="top10_retrievals.csv", top_k=10):

    records = []
    fig_dir = pathlib.Path("retrieval_visuals")
    fig_dir.mkdir(exist_ok=True)

    for i, query_embedding in enumerate(test_outputs['embeddings']):
        query_name = test_outputs['filenames'][i]
        query_tensor = normalize_embeddings(torch.tensor(query_embedding).unsqueeze(0).cuda())
        

        try:
            

            db_embeddings = normalize_embeddings(train_outputs['embeddings'])
            sims = torch.matmul(query_tensor, db_embeddings.T)[0]  # shape [N_train]
            topk_vals, topk_inds = torch.topk(sims, k=top_k)

            top_matches = [train_outputs['filenames'][i] for i in topk_inds.tolist()]
            scores = [topk_vals[i].item() for i in range(top_k)]

            records.append({
                "query": query_name,
                "top_k_matches": "|".join(top_matches),
                "scores": "|".join(f"{s:.4f}" for s in scores)
            })

            # Visualization Placeholder (1 query + top-10 text list)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.axis('off')
            text = f"Query: {query_name} " + " ".join([f"{i+1}. {fname} (score={scores[i]:.4f})" for i, fname in enumerate(top_matches)])
            ax.text(0.01, 1.0, text, fontsize=10, verticalalignment='top')
            fig_path = fig_dir / f"{query_name}_retrieval.png"
            plt.savefig(fig_path, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"❌ Retrieval or visualization error for query '{query_name}': {e}")

    # Save CSV
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved top-10 retrievals to {output_csv} and visuals to {fig_dir}/")
'''

def retrieve_single_sample(query_outputs, train_outputs, top_k=10):

    for i, query_embedding in enumerate(query_outputs['embeddings']):
        query_name = query_outputs['filenames'][i]
        query_tensor = normalize_embeddings(torch.tensor(query_embedding).unsqueeze(0))

        db_embeddings = normalize_embeddings(train_outputs['embeddings'])
        sims = torch.matmul(query_tensor, db_embeddings.T)[0]
        topk_vals, topk_inds = torch.topk(sims, k=top_k)

        top_matches = [train_outputs['filenames'][i] for i in topk_inds.tolist()]

        print(f"🔎 Query: {query_name} Top-{top_k} most similar CAD samples:")
        for j, fname in enumerate(top_matches):
            print(f"  {j+1}. {fname} (score: {topk_vals[j].item():.4f})")

    return top_matches



##############--------------Cosine Similarity Retrieval Function-----------------##############
def retrieve_top_k_cosine(query_outputs, db_outputs, top_k=5):
    start_time = time.time()

    query_embeds = query_outputs['embeddings']
    db_embeds = db_outputs['embeddings']

    # Convert to numpy if in tensor format
    if isinstance(query_embeds, torch.Tensor):
        query_embeds = query_embeds.cpu().numpy()
    if isinstance(db_embeds, torch.Tensor):
        db_embeds = db_embeds.cpu().numpy()

    sims = cosine_similarity(query_embeds, db_embeds)  # shape [N_query, N_db]

    topk_inds = np.argsort(-sims, axis=1)[:, :top_k]
    topk_vals = np.take_along_axis(sims, topk_inds, axis=1)

    retrieval_results = []
    correct_at_1 = 0
    correct_at_5 = 0
    y_true, y_pred, y_scores = [], [], []
    true_positives = 0

    for i, indices in enumerate(topk_inds):
        query_label = int(query_outputs['labels'][i])
        query_name = query_outputs['filenames'][i]
        top_labels = [int(db_outputs['labels'][j]) for j in indices]
        top_files = [db_outputs['filenames'][j] for j in indices]
        scores = topk_vals[i].tolist()

        match_at_1 = (top_labels[0] == query_label)
        match_at_5 = query_label in top_labels
        if match_at_1: correct_at_1 += 1
        if match_at_5: correct_at_5 += 1
        true_positives += sum(lbl == query_label for lbl in top_labels)

        y_true.append(query_label)
        y_pred.append(top_labels[0])
        y_scores.append(scores[0])

        retrieval_results.append({
            "query": query_name,
            "query_label": query_label,
            "top_k_files": top_files,
            "top_k_labels": top_labels,
            "match_flags": [int(l == query_label) for l in top_labels],
            "scores": scores,
        })

    total = len(query_outputs['labels'])
    recall_at_1 = correct_at_1 / total
    recall_at_5 = correct_at_5 / total
    top1_matches = [1 if y == p else 0 for y, p in zip(y_true, y_pred)]
    auc = roc_auc_score(top1_matches, y_scores) if len(set(top1_matches)) > 1 else None
    conf_mat = confusion_matrix(y_true, y_pred)

    elapsed = time.time() - start_time
    print(f"⏱ Retrieval with cosine similarity took {elapsed:.2f} seconds.")

    metrics = {
        "total_queries": total,
        "true_positives": true_positives,
        "recall@1": recall_at_1,
        "recall@5": recall_at_5,
        "auc_roc": auc,
        "retrieval_time_sec": elapsed
    }

    return retrieval_results, metrics, conf_mat

##############--------------Cosine Similarity Retrieval Function-----------------##############




##############--------------Retrieval Method: Learned Matrix-----------------##############
def retrieve_top_k_learned_matrix(query_outputs, db_outputs, top_k=5, projection_matrix=None):
    start_time = time.time()

    query_embeds = query_outputs['embeddings']
    db_embeds = db_outputs['embeddings']

    if isinstance(query_embeds, torch.Tensor):
        query_embeds = query_embeds.cpu().numpy()
    if isinstance(db_embeds, torch.Tensor):
        db_embeds = db_embeds.cpu().numpy()

    if projection_matrix is None:
        projection_matrix = np.eye(query_embeds.shape[1])  # Identity as fallback

    query_projected = query_embeds @ projection_matrix
    db_projected = db_embeds @ projection_matrix

    sims = cosine_similarity(query_projected, db_projected)
    topk_inds = np.argsort(-sims, axis=1)[:, :top_k]
    topk_vals = np.take_along_axis(sims, topk_inds, axis=1)

    retrieval_results, correct_at_1, correct_at_5, y_true, y_pred, y_scores = [], 0, 0, [], [], []
    true_positives = 0

    for i, indices in enumerate(topk_inds):
        query_label = int(query_outputs['labels'][i])
        query_name = query_outputs['filenames'][i]
        top_labels = [int(db_outputs['labels'][j]) for j in indices]
        top_files = [db_outputs['filenames'][j] for j in indices]
        scores = topk_vals[i].tolist()

        match_at_1 = (top_labels[0] == query_label)
        match_at_5 = query_label in top_labels
        if match_at_1: correct_at_1 += 1
        if match_at_5: correct_at_5 += 1
        true_positives += sum(lbl == query_label for lbl in top_labels)

        y_true.append(query_label)
        y_pred.append(top_labels[0])
        y_scores.append(scores[0])

        retrieval_results.append({"query": query_name, 
                                  "query_label": query_label, 
                                  "top_k_files": top_files, 
                                  "top_k_labels": top_labels, 
                                  "match_flags": [int(l == query_label) for l in top_labels], 
                                  "scores": scores})

    total = len(query_outputs['labels'])
    recall_at_1 = correct_at_1 / total
    recall_at_5 = correct_at_5 / total
    top1_matches = [1 if y == p else 0 for y, p in zip(y_true, y_pred)]
    auc = roc_auc_score(top1_matches, y_scores) if len(set(top1_matches)) > 1 else None
    conf_mat = confusion_matrix(y_true, y_pred)
    elapsed = time.time() - start_time
    print(f"⏱ Retrieval with learned matrix took {elapsed:.2f} seconds.")

    metrics = {"total_queries": total, 
               "true_positives": true_positives, 
               "recall@1": recall_at_1, 
               "recall@5": recall_at_5, 
               "auc_roc": auc, 
               "retrieval_time_sec": elapsed}

    return retrieval_results, metrics, conf_mat

##############--------------Retrieval Method: Learned Matrix-----------------##############



##############--------------L2/Euclidean Distance-----------------##############

def retrieve_top_k_l2(query_outputs, db_outputs, top_k=5):
    start_time = time.time()

    query_embeds = query_outputs['embeddings']
    db_embeds = db_outputs['embeddings']

    if isinstance(query_embeds, torch.Tensor):
        query_embeds = query_embeds.cpu().numpy()
    if isinstance(db_embeds, torch.Tensor):
        db_embeds = db_embeds.cpu().numpy()

    dists = euclidean_distances(query_embeds, db_embeds)
    topk_inds = np.argsort(dists, axis=1)[:, :top_k]
    topk_vals = np.take_along_axis(dists, topk_inds, axis=1)

    retrieval_results = []
    correct_at_1 = 0
    correct_at_5 = 0
    y_true, y_pred, y_scores = [], [], []
    true_positives = 0

    for i, indices in enumerate(topk_inds):
        query_label = int(query_outputs['labels'][i])
        query_name = query_outputs['filenames'][i]
        top_labels = [int(db_outputs['labels'][j]) for j in indices]
        top_files = [db_outputs['filenames'][j] for j in indices]
        scores = topk_vals[i].tolist()

        match_at_1 = (top_labels[0] == query_label)
        match_at_5 = query_label in top_labels
        if match_at_1: correct_at_1 += 1
        if match_at_5: correct_at_5 += 1
        true_positives += sum(lbl == query_label for lbl in top_labels)

        y_true.append(query_label)
        y_pred.append(top_labels[0])
        y_scores.append(-scores[0])

        retrieval_results.append({"query": query_name, 
                                  "query_label": query_label, 
                                  "top_k_files": top_files, 
                                  "top_k_labels": top_labels, 
                                  "match_flags": [int(l == query_label) for l in top_labels], 
                                  "scores": scores})

    total = len(query_outputs['labels'])
    recall_at_1 = correct_at_1 / total
    recall_at_5 = correct_at_5 / total
    top1_matches = [1 if y == p else 0 for y, p in zip(y_true, y_pred)]
    auc = roc_auc_score(top1_matches, y_scores) if len(set(top1_matches)) > 1 else None
    conf_mat = confusion_matrix(y_true, y_pred)

    elapsed = time.time() - start_time
    print(f"⏱ Retrieval with Euclidean distance took {elapsed:.2f} seconds.")

    metrics = {"total_queries": total, 
               "true_positives": true_positives, 
               "recall@1": recall_at_1, 
               "recall@5": recall_at_5, 
               "auc_roc": auc, 
               "retrieval_time_sec": elapsed}

    return retrieval_results, metrics, conf_mat

##############--------------L2/Euclidean Distance-----------------##############


##############--------------Hierarchical Navigable Small World graphs-----------------##############
def retrieve_top_k_hnsw(query_outputs, db_outputs, top_k=5):
    start_time = time.time()

    query_embeds = query_outputs['embeddings']
    db_embeds = db_outputs['embeddings']

    if isinstance(query_embeds, torch.Tensor):
        query_embeds = query_embeds.cpu().numpy()
    if isinstance(db_embeds, torch.Tensor):
        db_embeds = db_embeds.cpu().numpy()
    
    dim = query_embeds.shape[1]
    index = hnswlib.Index(space='cosine', dim=dim)

    index.init_index(max_elements=len(db_embeds), ef_construction=100, M=16)
    index.add_items(db_embeds)
    index.set_ef(top_k + 10)
    topk_inds, topk_dists = index.knn_query(query_embeds, k=top_k)

    retrieval_results = []
    correct_at_1 = 0
    correct_at_5 = 0
    y_true, y_pred, y_scores = [], [], []
    true_positives = 0

    for i, indices in enumerate(topk_inds):
        query_label = int(query_outputs['labels'][i])
        query_name = query_outputs['filenames'][i]
        top_labels = [int(db_outputs['labels'][j]) for j in indices]
        top_files = [db_outputs['filenames'][j] for j in indices]
        scores = (1 - topk_dists[i]).tolist()

        match_at_1 = (top_labels[0] == query_label)
        match_at_5 = query_label in top_labels
        if match_at_1: correct_at_1 += 1
        if match_at_5: correct_at_5 += 1
        true_positives += sum(lbl == query_label for lbl in top_labels)

        y_true.append(query_label)
        y_pred.append(top_labels[0])
        y_scores.append(scores[0])

        retrieval_results.append({"query": query_name, 
                                  "query_label": query_label, 
                                  "top_k_files": top_files, 
                                  "top_k_labels": top_labels, 
                                  "match_flags": [int(l == query_label) for l in top_labels], 
                                  "scores": scores})

    total = len(query_outputs['labels'])
    recall_at_1 = correct_at_1 / total
    recall_at_5 = correct_at_5 / total
    top1_matches = [1 if y == p else 0 for y, p in zip(y_true, y_pred)]
    auc = roc_auc_score(top1_matches, y_scores) if len(set(top1_matches)) > 1 else None
    conf_mat = confusion_matrix(y_true, y_pred)

    elapsed = time.time() - start_time
    print(f"⏱ Retrieval with HNSW took {elapsed:.2f} seconds.")

    metrics = {"total_queries": total, 
               "true_positives": true_positives, 
               "recall@1": recall_at_1, 
               "recall@5": recall_at_5, 
               "auc_roc": auc, 
               "retrieval_time_sec": elapsed}

    return retrieval_results, metrics, conf_mat

##############--------------Hierarchical Navigable Small World graphs-----------------##############




##############--------------Approximate Nearest Neighbor (ANN) Search-----------------##############
def retrieve_top_k_faiss(query_outputs, db_outputs, top_k=5):
    start_time = time.time()

    query_embeds = query_outputs['embeddings']
    db_embeds = db_outputs['embeddings']

    if isinstance(query_embeds, torch.Tensor):
        query_embeds = query_embeds.cpu().numpy().astype(np.float32)
    if isinstance(db_embeds, torch.Tensor):
        db_embeds = db_embeds.cpu().numpy().astype(np.float32)

    dim = db_embeds.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(db_embeds)
    topk_dists, topk_inds = index.search(query_embeds, top_k)

    retrieval_results = []
    correct_at_1 = 0
    correct_at_5 = 0
    y_true, y_pred, y_scores = [], [], []
    true_positives = 0

    for i, indices in enumerate(topk_inds):
        query_label = int(query_outputs['labels'][i])
        query_name = query_outputs['filenames'][i]
        top_labels = [int(db_outputs['labels'][j]) for j in indices]
        top_files = [db_outputs['filenames'][j] for j in indices]
        scores = (-topk_dists[i]).tolist()

        match_at_1 = (top_labels[0] == query_label)
        match_at_5 = query_label in top_labels
        if match_at_1: correct_at_1 += 1
        if match_at_5: correct_at_5 += 1
        true_positives += sum(lbl == query_label for lbl in top_labels)

        y_true.append(query_label)
        y_pred.append(top_labels[0])
        y_scores.append(scores[0])

        retrieval_results.append({"query": query_name, 
                                  "query_label": query_label, 
                                  "top_k_files": top_files, 
                                  "top_k_labels": top_labels, 
                                  "match_flags": [int(l == query_label) for l in top_labels], 
                                  "scores": scores})

    total = len(query_outputs['labels'])
    recall_at_1 = correct_at_1 / total
    recall_at_5 = correct_at_5 / total
    top1_matches = [1 if y == p else 0 for y, p in zip(y_true, y_pred)]
    auc = roc_auc_score(top1_matches, y_scores) if len(set(top1_matches)) > 1 else None
    conf_mat = confusion_matrix(y_true, y_pred)

    elapsed = time.time() - start_time
    print(f"⏱ Retrieval with FAISS L2 took {elapsed:.2f} seconds.")

    metrics = {"total_queries": total, 
               "true_positives": true_positives, 
               "recall@1": recall_at_1, 
               "recall@5": recall_at_5, 
               "auc_roc": auc, 
               "retrieval_time_sec": elapsed}

    return retrieval_results, metrics, conf_mat
##############--------------Approximate Nearest Neighbor (ANN) Search-----------------##############



##############--------------Autoencoder Method-----------------##############
def retrieve_top_k_autoencoder(query_outputs, db_outputs, top_k=5):
    start_time = time.time()

    query_embeds = query_outputs['embeddings']
    db_embeds = db_outputs['embeddings']

    if isinstance(query_embeds, torch.Tensor):
        query_embeds = query_embeds.cpu().numpy()
    if isinstance(db_embeds, torch.Tensor):
        db_embeds = db_embeds.cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    all_embeds = np.vstack([query_embeds, db_embeds])
    tsne_proj = tsne.fit_transform(all_embeds)
    query_proj = tsne_proj[:len(query_embeds)]
    db_proj = tsne_proj[len(query_embeds):]

    nbrs = NearestNeighbors(n_neighbors=top_k, metric='euclidean').fit(db_proj)
    topk_vals, topk_inds = nbrs.kneighbors(query_proj)

    retrieval_results = []
    correct_at_1 = 0
    correct_at_5 = 0
    y_true, y_pred, y_scores = [], [], []
    true_positives = 0

    for i, indices in enumerate(topk_inds):
        query_label = int(query_outputs['labels'][i])
        query_name = query_outputs['filenames'][i]
        top_labels = [int(db_outputs['labels'][j]) for j in indices]
        top_files = [db_outputs['filenames'][j] for j in indices]
        scores = [-topk_vals[i][j] for j in range(top_k)]

        match_at_1 = (top_labels[0] == query_label)
        match_at_5 = query_label in top_labels
        if match_at_1: correct_at_1 += 1
        if match_at_5: correct_at_5 += 1
        true_positives += sum(lbl == query_label for lbl in top_labels)

        y_true.append(query_label)
        y_pred.append(top_labels[0])
        y_scores.append(scores[0])

        retrieval_results.append({"query": query_name, 
                                  "query_label": query_label, 
                                  "top_k_files": top_files, 
                                  "top_k_labels": top_labels, 
                                  "match_flags": [int(l == query_label) for l in top_labels], 
                                  "scores": scores})

    total = len(query_outputs['labels'])
    recall_at_1 = correct_at_1 / total
    recall_at_5 = correct_at_5 / total
    top1_matches = [1 if y == p else 0 for y, p in zip(y_true, y_pred)]
    auc = roc_auc_score(top1_matches, y_scores) if len(set(top1_matches)) > 1 else None
    conf_mat = confusion_matrix(y_true, y_pred)
    elapsed = time.time() - start_time
    print(f"⏱ Autoencoder-style retrieval (TSNE projection) took {elapsed:.2f} seconds.")

    metrics = {"total_queries": total, 
               "true_positives": true_positives, 
               "recall@1": recall_at_1, 
               "recall@5": recall_at_5, 
               "auc_roc": auc, 
               "retrieval_time_sec": elapsed}
    return retrieval_results, metrics, conf_mat

    ##############--------------Autoencoder Method-----------------##############
