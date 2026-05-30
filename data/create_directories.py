from config.settings import settings


def create_project_directories():

    directories = [
        settings.DATA_DIR,
        settings.RAW_DATA_DIR,
        settings.PROCESSED_DATA_DIR,
        settings.EMBEDDINGS_DIR,
        settings.LOGS_DIR,
    ]

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    print("Project directories initialized.")


if __name__ == "__main__":

    create_project_directories()