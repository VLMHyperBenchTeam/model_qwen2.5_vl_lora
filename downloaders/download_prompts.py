import gdown

if __name__ == '__main__':
    prompts_foolder_id = "1kM2zsRgPB2RaXvbwmvM9PHAvlvClR1PW"
    google_drive_url = 'https://drive.google.com/drive/folders/'
    prompts_foolder_url = google_drive_url + prompts_foolder_id
    gdown.download_folder(prompts_foolder_url)