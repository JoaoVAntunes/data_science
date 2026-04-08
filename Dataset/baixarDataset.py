import kagglehub

# Download latest version
path = kagglehub.dataset_download("corrieaar/apartment-rental-offers-in-germany")

print("Path to dataset files:", path)