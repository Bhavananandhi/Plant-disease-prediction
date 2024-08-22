import gdown

# Define the file ID
file_id = "1-025nCEuv7YsogDwFMNAf8ONrpUdYgZ7"

# Construct the full URL to the file
url = f"https://drive.google.com/uc?id={file_id}"

# Download the file
output = "plant_disease_prediction_model.h5"
gdown.download(url, output, quiet=False)
