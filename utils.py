from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, SpectralClustering, MeanShift, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
import numpy as np
import torch
from scipy.stats import mode
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap
import seaborn as sns

def perform_clustering(model, dataset, latent_dim=128):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    latents = []
    true_labels = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            _, z, *vae_params = model(inputs)
            latents.append(z.cpu().numpy())
            true_labels.extend(labels)
    latents = np.vstack(latents)
    true_labels = np.array(true_labels)

    clustering_methods = [
        ('K-Means', KMeans(n_clusters=8, random_state=42)),
        ('GMM', GaussianMixture(n_components=8, random_state=42)),
        ('DBSCAN', DBSCAN(eps=0.5, min_samples=5)),
        ('Agglomerative', AgglomerativeClustering(n_clusters=8)),
        ('Spectral', SpectralClustering(n_clusters=8, random_state=42, affinity='nearest_neighbors')),
        ('MeanShift', MeanShift(bandwidth=2.0)),
        ('Birch', Birch(n_clusters=8))
    ]
    
    results = {}
    best_method = None
    best_ari = -1
    
    for name, method in clustering_methods:
        try:
            print(f"Running {name} clustering...")
            if name == 'DBSCAN' or name == 'MeanShift':
                labels = method.fit_predict(latents)
                if len(set(labels)) <= 1 or (name == 'DBSCAN' and np.sum(labels == -1) > 0.5 * len(labels)):
                    print(f"Skipping {name}: insufficient clustering")
                    continue
            else:
                labels = method.fit_predict(latents)
            
            metrics = {}
            try:
                metrics['silhouette'] = silhouette_score(latents, labels)
            except:
                metrics['silhouette'] = float('nan')
                
            try:
                metrics['davies_bouldin'] = davies_bouldin_score(latents, labels)
            except:
                metrics['davies_bouldin'] = float('nan')
                
            metrics['ari'] = adjusted_rand_score(true_labels, labels)
            
            cluster_to_genre = {}
            for cluster in set(labels):
                if cluster == -1: 
                    continue
                mask = (labels == cluster)
                unique_values, counts = np.unique(true_labels[mask], return_counts=True)
                most_common = unique_values[np.argmax(counts)] if len(counts) > 0 else None
                cluster_to_genre[cluster] = most_common
            
            pred_labels = []
            for label in labels:
                if label == -1 and name == 'DBSCAN':
                    pred_labels.append('noise')
                else:
                    pred_labels.append(cluster_to_genre.get(label, true_labels[0]))
            
            print(f"{name} Silhouette Score: {metrics['silhouette']:.4f}, Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}, ARI: {metrics['ari']:.4f}")
            print(f"{name} Classification Report:")
            print(classification_report(true_labels, pred_labels, zero_division=0))
            
            results[name] = {
                'labels': labels,
                'metrics': metrics,
                'cluster_to_genre': cluster_to_genre,
                'pred_labels': pred_labels
            }
            
            if metrics['ari'] > best_ari:
                best_ari = metrics['ari']
                best_method = name
                
        except Exception as e:
            print(f"Error with {name} clustering: {e}")
    
    compare_clustering_methods(results, true_labels)
    
    visualize_all_clustering_methods(latents, results, true_labels, dataset)
    
    if 'K-Means' in results:
        kmeans_labels = results['K-Means']['labels']
    else:
        kmeans = KMeans(n_clusters=10, random_state=42)
        kmeans_labels = kmeans.fit_predict(latents)
    
    return latents, kmeans_labels, true_labels, results

def compare_clustering_methods(results, true_labels):
    """Create a comparison table of all clustering methods"""
    if not results:
        print("No clustering results to compare")
        return
        
    plt.figure(figsize=(14, 10))
    
    metrics = ['silhouette', 'davies_bouldin', 'ari']
    metric_names = ['Silhouette Score', 'Davies-Bouldin Index', 'Adjusted Rand Index']
    colors = ['#2ca02c', '#d62728', '#1f77b4']
    
    methods = list(results.keys())
    data = {metric: [results[method]['metrics'].get(metric, float('nan')) for method in methods] 
            for metric in metrics}
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        plt.subplot(3, 1, i+1)
        bars = plt.bar(methods, data[metric], color=colors[i])
        plt.title(name, fontsize=12, pad=10)
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(hspace=0.4)
        
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout(pad=2.0)
    plt.savefig('clustering_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')
    ax.axis('tight')
    
    table_data = []
    for method in methods:
        row = [method]
        for metric in metrics:
            value = results[method]['metrics'].get(metric, float('nan'))
            row.append(f"{value:.4f}")
        table_data.append(row)
    
    table = ax.table(cellText=table_data, 
                    colLabels=['Method'] + metric_names,
                    loc='center', cellLoc='center',
                    colWidths=[0.25] + [0.25] * len(metrics))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8) 
    
    for i, metric in enumerate(metrics):
        col_data = [results[method]['metrics'].get(metric, float('nan')) for method in methods]
        if metric == 'davies_bouldin':
            best_idx = np.nanargmin(col_data)
        else:
            best_idx = np.nanargmax(col_data)
        
        if not np.isnan(col_data[best_idx]):
            table[(best_idx+1, i+1)].set_facecolor('#90ee90')
    
    plt.title('Clustering Methods Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout(pad=1.5)
    plt.savefig('clustering_comparison_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Clustering comparison saved to 'clustering_comparison.png' and 'clustering_comparison_table.png'")

def visualize_results(latents, kmeans_labels, true_labels, dataset):
    tsne = TSNE(n_components=2, random_state=42)
    latents_2d = tsne.fit_transform(latents)

    plt.figure(figsize=(16, 7), dpi=300)
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=kmeans_labels,
                         cmap='tab10', s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    plt.colorbar(scatter, label='Cluster ID', ticks=range(len(np.unique(kmeans_labels))))
    plt.title('Clustering Results (t-SNE)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    plt.ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    
    plt.subplot(1, 2, 2)
    genre_to_idx = {genre: i for i, genre in enumerate(sorted(set(true_labels)))}
    genre_indices = np.array([genre_to_idx[label] for label in true_labels])
    
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=genre_indices,
                         cmap='tab10', s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab10(idx/len(genre_to_idx)),
                                 markersize=10, label=genre.capitalize())
                      for genre, idx in genre_to_idx.items()]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
              title='Genres', title_fontsize=12, fontsize=10)
    
    plt.title('True Genre Distribution (t-SNE)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    plt.ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('latent_space.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(15, 10))
    
    unique_clusters = np.unique(kmeans_labels)
    n_clusters = len(unique_clusters)
    
    for i, cluster in enumerate(unique_clusters):
        if i >= 10:
            break
            
        indices = np.where(kmeans_labels == cluster)[0]
        if len(indices) == 0:
            continue
            
        idx = indices[0]
        image, label = dataset[idx]
        
        plt.subplot(2, 5, i + 1)
        
        if image.shape[0] == 3:
            img_data = image.permute(1, 2, 0).numpy()
        else:
            img_data = image.squeeze().numpy()
        
        plt.imshow(img_data, cmap='viridis' if image.shape[0] == 1 else None)
        
        cluster_indices = np.where(kmeans_labels == cluster)[0]
        cluster_labels = [true_labels[i] for i in cluster_indices]
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        most_common = unique_labels[np.argmax(counts)]
        
        plt.title(f'Cluster {cluster}\nMost common: {most_common}\n({len(cluster_indices)} samples)', 
                  fontsize=10)
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig('cluster_spectrograms.png', dpi=300, bbox_inches='tight')
    plt.close()

def perform_eda(dataset):
    expected_genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    genres = [label for _, label in dataset]
    
    genre_counts = {}
    for genre in genres:
        if genre in genre_counts:
            genre_counts[genre] += 1
        else:
            genre_counts[genre] = 1
    
    missing_genres = [genre for genre in expected_genres if genre not in genre_counts]
    if missing_genres:
        print(f"Warning: Missing genres in dataset: {missing_genres}")
        for genre in missing_genres:
            genre_counts[genre] = 0
    
    plt.figure(figsize=(12, 8), dpi=300)
    sorted_genres = sorted(genre_counts.keys())
    counts = [genre_counts[genre] for genre in sorted_genres]
    
    bars = plt.bar(sorted_genres, counts, edgecolor='black', color='#2196F3', alpha=0.7)
    plt.title('Genre Distribution in Dataset', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Music Genre', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', va='bottom', fontsize=11)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('genre_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, axs = plt.subplots(2, 5, figsize=(20, 8), dpi=300)
    fig.suptitle('Sample Spectrograms by Genre', fontsize=16, fontweight='bold', y=1.02)
    
    for i, genre in enumerate(expected_genres):
        if i >= 10:
            break
            
        valid_image = None
        for idx in range(len(dataset)):
            try:
                image, label = dataset[idx]
                if label == genre and image is not None and torch.isfinite(image).all():
                    valid_image = image
                    break
            except Exception as e:
                print(f"Error loading image at index {idx}: {e}")
                continue
        
        if valid_image is not None:
            ax = axs[i // 5, i % 5]
            
            try:
                if valid_image.shape[0] == 3:
                    img_data = valid_image.permute(1, 2, 0).numpy()
                else:
                    img_data = valid_image.squeeze().numpy()
                
                if np.isfinite(img_data).all():
                    im = ax.imshow(img_data, cmap='viridis')
                    ax.set_title(genre.capitalize(), fontsize=12, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, f"{genre}\n(Invalid data)", 
                            horizontalalignment='center',
                            verticalalignment='center')
            except Exception as e:
                print(f"Error displaying image for genre {genre}: {e}")
                ax.text(0.5, 0.5, f"{genre}\n(Error)", 
                        horizontalalignment='center',
                        verticalalignment='center')
            
            ax.axis('off')
    
    fig.colorbar(im, ax=axs.ravel().tolist(), label='Magnitude (dB)', shrink=0.8)
    plt.tight_layout()
    plt.savefig('sample_spectrograms.png', dpi=300, bbox_inches='tight')
    plt.close()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_confusion_matrix(true_labels, pred_labels, class_names=None, normalize=True):
    plt.figure(figsize=(12, 10), dpi=300)
    
    cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', square=True, linewidths=0.5,
                xticklabels=[x.capitalize() for x in class_names],
                yticklabels=[x.capitalize() for x in class_names])
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_curves(train_losses, val_losses):
    plt.figure(figsize=(12, 8), dpi=300)
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.plot(epochs, train_losses, 'bo', markersize=4)
    plt.plot(epochs, val_losses, 'ro', markersize=4)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Training and Validation Loss Curves', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    
    plt.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='black',
              loc='upper right')
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.xlim(0.8, len(train_losses) + 0.2)
    y_min = min(min(train_losses), min(val_losses)) * 0.95
    y_max = max(max(train_losses), max(val_losses)) * 1.05
    plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_all_clustering_methods(latents, results, true_labels, dataset):
    """
    Visualize each clustering algorithm's results individually and save separate images
    """
    # t-SNE for dimensionality reduction (compute once for consistency)
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42)
    latents_2d = tsne.fit_transform(latents)
    
    # Create genre to index mapping for consistent colors
    unique_genres = sorted(set(true_labels))
    genre_to_idx = {genre: i for i, genre in enumerate(unique_genres)}
    
    # First, create a reference plot of true genres
    plt.figure(figsize=(10, 8))
    genre_indices = np.array([genre_to_idx[label] for label in true_labels])
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=genre_indices, 
                         cmap='tab10', s=30, alpha=0.8, edgecolors='w', linewidths=0.5)
    
    # Add legend for genres
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=plt.cm.tab10(genre_to_idx[genre]/len(genre_to_idx)), 
                                 markersize=8, label=genre) 
                      for genre in unique_genres]
    plt.legend(handles=legend_elements, loc='best', title='Genres', 
               fontsize=9, title_fontsize=10)
    
    plt.title('True Genre Distribution (t-SNE)', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE dimension 1', fontsize=12)
    plt.ylabel('t-SNE dimension 2', fontsize=12)
    plt.tight_layout()
    plt.savefig('true_genres_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the original data points with no clustering (just colored by genre)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=genre_indices, 
                         cmap='tab10', s=30, alpha=0.8, edgecolors='w', linewidths=0.5)
    plt.title('Original Data Distribution (No Clustering)', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE dimension 1', fontsize=12)
    plt.ylabel('t-SNE dimension 2', fontsize=12)
    plt.legend(handles=legend_elements, loc='best', title='Genres', 
              fontsize=9, title_fontsize=10)
    plt.tight_layout()
    plt.savefig('original_data_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Now visualize each clustering method
    for method_name, result in results.items():
        print(f"Visualizing {method_name} clustering...")
        labels = result['labels']
        
        # Skip methods that failed
        if labels is None:
            continue
            
        # Create a 2x2 subplot figure to include original data for comparison
        fig, axs = plt.subplots(2, 2, figsize=(16, 14))
        
        # Plot 1: Original data distribution (top left)
        scatter = axs[0, 0].scatter(latents_2d[:, 0], latents_2d[:, 1], c=genre_indices, 
                                  cmap='tab10', s=30, alpha=0.8, edgecolors='w', linewidths=0.5)
        axs[0, 0].set_title('Original Data (True Genres)', fontsize=14, fontweight='bold')
        axs[0, 0].set_xlabel('t-SNE dimension 1', fontsize=12)
        axs[0, 0].set_ylabel('t-SNE dimension 2', fontsize=12)
        
        # Add legend for genres
        legend1 = axs[0, 0].legend(handles=legend_elements, loc='best', title='Genres', 
                                 fontsize=9, title_fontsize=10)
        
        # Plot 2: Cluster assignments (top right)
        n_clusters = len(np.unique(labels))
        scatter = axs[0, 1].scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, 
                                  cmap='tab10', s=30, alpha=0.8, edgecolors='w', linewidths=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axs[0, 1], label='Cluster ID')
        
        # For DBSCAN, handle -1 noise points specially
        if method_name == 'DBSCAN' and -1 in np.unique(labels):
            # Adjust colorbar ticks to show "noise" for -1
            ticks = np.unique(labels)
            cbar.set_ticks(ticks)
            tick_labels = [str(t) if t != -1 else 'noise' for t in ticks]
            cbar.set_ticklabels(tick_labels)
        
        axs[0, 1].set_title(f'{method_name} Clusters', fontsize=14, fontweight='bold')
        axs[0, 1].set_xlabel('t-SNE dimension 1', fontsize=12)
        axs[0, 1].set_ylabel('t-SNE dimension 2', fontsize=12)
        
        # Plot 3: Confusion visualization (bottom left)
        pred_labels = result['pred_labels']
        
        # Create a scatter plot with different markers for different clusters
        # Use only filled markers to avoid warnings with edgecolors
        filled_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H']
        
        # Get unique cluster labels
        unique_clusters = np.unique(labels)
        
        # For each cluster, plot points with the same marker but colored by true genre
        for i, cluster in enumerate(unique_clusters):
            if cluster == -1 and method_name == 'DBSCAN':  # Special handling for DBSCAN noise
                mask = (labels == cluster)
                # For noise points, don't use edgecolor
                axs[1, 0].scatter(latents_2d[mask, 0], latents_2d[mask, 1], 
                                c='gray', marker='x', s=20, alpha=0.5, label='noise')
            else:
                mask = (labels == cluster)
                marker = filled_markers[i % len(filled_markers)]  # Cycle through filled markers only
                
                # Get true genre indices for coloring
                genre_indices_cluster = np.array([genre_to_idx[true_labels[j]] for j in np.where(mask)[0]])
                
                # Plot with consistent colors from genre_to_idx
                for genre in unique_genres:
                    genre_mask = (true_labels[mask] == genre)
                    if np.any(genre_mask):
                        idx = genre_to_idx[genre]
                        color = plt.cm.tab10(idx / len(genre_to_idx))
                        axs[1, 0].scatter(latents_2d[mask][genre_mask, 0], 
                                        latents_2d[mask][genre_mask, 1],
                                        color=color, marker=marker, s=30, alpha=0.7,
                                        edgecolors='w', linewidths=0.5)
        
        # Add a title explaining the visualization
        axs[1, 0].set_title(f'Genre-Cluster Relationship', fontsize=14, fontweight='bold')
        axs[1, 0].set_xlabel('t-SNE dimension 1', fontsize=12)
        axs[1, 0].set_ylabel('t-SNE dimension 2', fontsize=12)
        axs[1, 0].legend(handles=legend_elements, loc='best', title='Genres', 
                       fontsize=9, title_fontsize=10)
        
        # Plot 4: Confusion matrix (bottom right)
        import seaborn as sns
        
        # Get unique classes
        class_names = sorted(set(true_labels))
        
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
        
        # Normalize
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, 
                   linewidths=0.5, linecolor='black', square=True, ax=axs[1, 1])
        
        axs[1, 1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axs[1, 1].set_xlabel('Predicted label', fontsize=12)
        axs[1, 1].set_ylabel('True label', fontsize=12)
        axs[1, 1].set_xticklabels(axs[1, 1].get_xticklabels(), rotation=45, ha='right')
        
        # Add overall metrics as text
        metrics = result['metrics']
        metrics_text = (
            f"Silhouette Score: {metrics.get('silhouette', 'N/A'):.4f}\n"
            f"Davies-Bouldin Index: {metrics.get('davies_bouldin', 'N/A'):.4f}\n"
            f"Adjusted Rand Index: {metrics.get('ari', 'N/A'):.4f}"
        )
        axs[1, 1].annotate(metrics_text, xy=(0.5, -0.15), xycoords='axes fraction', 
                         ha='center', va='center', fontsize=10,
                         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(f'{method_name}_clustering_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save individual plots for each clustering method
        # Cluster assignments only
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, 
                             cmap='tab10', s=30, alpha=0.8, edgecolors='w', linewidths=0.5)
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(f'{method_name} Clustering Result', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE dimension 1', fontsize=12)
        plt.ylabel('t-SNE dimension 2', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{method_name}_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a confusion matrix for each method
        target_filename = f'{method_name}_confusion_matrix.png'
        
        # Remove the target file if it already exists
        import os
        if os.path.exists(target_filename):
            os.remove(target_filename)
        
        # Create the confusion matrix
        plot_confusion_matrix(true_labels, pred_labels, 
                             class_names=sorted(set(true_labels)), 
                             normalize=True)
        
        # Rename the saved confusion matrix to method-specific name
        if os.path.exists('confusion_matrix.png'):
            os.rename('confusion_matrix.png', target_filename)
            print(f"Saved confusion matrix for {method_name}")