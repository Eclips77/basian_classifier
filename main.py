from src.app_controller import AppController
from services.file_loader import FileLoader

LABEL_COL = "BoughtComputer"


def main() -> None:
    """Demonstrate how to use ``AppController``."""
    loader = FileLoader()
    controller = AppController(LABEL_COL, loader=loader)

    data_path = FileLoader.select_data_file()
    if not data_path:
        print("No valid file selected. Exiting.")
        return

    controller.load_and_prepare(data_path)
    controller.train_model()

    accuracy = controller.get_accuracy()
    print(f"Model accuracy: {accuracy:.2%}")

    schema = controller.get_schema()
    print("Schema:")
    for feature, options in schema.items():
        print(f"  {feature}: {options}")

    sample_record = {feat: opts[0] for feat, opts in schema.items() if opts}
    prediction = controller.predict_record(sample_record)
    print(f"Prediction for {sample_record}: {prediction}")


if __name__ == "__main__":
    main()
