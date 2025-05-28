import gdown

if __name__ == '__main__':
    file_id = "1IBPr80SmPGmcGIQyIKgdaRhedu8h1Rbg"
    output = "dataset_for_training.zip"
    gdown.download(id=file_id, output=output, quiet=False)