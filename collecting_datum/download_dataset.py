import gdown


filename = 'https://drive.google.com/file/d/1MjwKFpPSqTQqYQ62PHPc8eZXQ7Bi0BbX/view?usp=sharing'


# URL of the file to download

# Path where you want to save the downloaded file
output_path = 'your_output_path/filename'

# Download the file
gdown.download(url, output_path, quiet=False)