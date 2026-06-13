from app.api import create_app
app = create_app(model_path="./outputs/grpo_lora")
if __name__ == "__main__":
    app.run()