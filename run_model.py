import torch
from data_setup import GTZANDataset
from utils import perform_eda, count_parameters, perform_clustering, visualize_results, plot_confusion_matrix, plot_loss_curves
from torch.utils.data import DataLoader, random_split
from model_builder import ConvVAE  # Updated to VAE
import numpy as np
from train import train_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root_dir = './Data/images_original'
dataset = GTZANDataset(root_dir, train=True)
val_dataset = GTZANDataset(root_dir, train=False)
num_channels = dataset.get_num_channels()

perform_eda(dataset)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

sample_image, _ = dataset[0]
input_size = (sample_image.shape[1], sample_image.shape[2])

latent_dim = 256
model = ConvVAE(latent_dim=latent_dim, in_channels=num_channels, input_size=input_size).to(device)
print(f"Total trainable parameters: {count_parameters(model)}")

training_losses, validation_losses = train_model(model, train_loader, val_loader, num_epochs=100, lr=0.0001, device=device)
model.load_state_dict(torch.load('best_model.pth'))

latents, kmeans_labels, true_labels, clustering_results = perform_clustering(model, dataset, latent_dim)
visualize_results(latents, kmeans_labels, true_labels, dataset)

true_labels_array = np.array(true_labels)
genre_predictions = []
for label in kmeans_labels:
    mask = (kmeans_labels == label)
    unique_values, counts = np.unique(true_labels_array[mask], return_counts=True)
    most_common = unique_values[np.argmax(counts)] if len(counts) > 0 else true_labels[0]
    genre_predictions.append(most_common)
plot_confusion_matrix(true_labels, genre_predictions, class_names=sorted(set(true_labels)))

plot_loss_curves(training_losses, validation_losses)