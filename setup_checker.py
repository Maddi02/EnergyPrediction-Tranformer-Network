import importlib
import subprocess
import sys
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

def check_library(lib_name):
    try:
        importlib.import_module(lib_name)
        print(f"{lib_name}: ✅ Installed")
        return True
    except ImportError:
        print(f"{lib_name}: ❌ Not Installed")
        return False

def install_library(lib_name):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib_name])
        print(f"{lib_name}: ✅ Successfully Installed")
    except Exception as e:
        print(f"{lib_name}: ❌ Installation Failed - {e}")

def check_pytorch():
    try:
        print(f"PyTorch version: {torch.__version__}")
        print("PyTorch: ✅ Setup correctly")
    except Exception as e:
        print(f"PyTorch: ❌ Error - {e}")

def check_gpu():
    if torch.cuda.is_available():
        print("PyTorch: ✅ GPU is available")
    else:
        print("PyTorch: ❌ GPU not available")

def check_transformers():
    try:
        classifier = pipeline("sentiment-analysis")
        print("Transformers pipeline test: ✅ Success")
        result = classifier("I love programming!")
        print(f"Sample result: {result}")
    except Exception as e:
        print(f"Transformers: ❌ Error - {e}")

def check_model(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        print(f"Model '{model_name}': ✅ Loaded Successfully")
        classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        test_result = classifier("This is awesome!")
        print(f"Test Inference with '{model_name}': {test_result}")
    except Exception as e:
        print(f"Model '{model_name}': ❌ Loading Failed - {e}")

if __name__ == "__main__":
    print("Checking Libraries...\n")
    pytorch_installed = check_library("torch")
    transformers_installed = check_library("transformers")

    print("\nDetailed Checks:\n")
    if pytorch_installed:
        check_pytorch()
        check_gpu()
    else:
        install_library("torch")
        check_pytorch()
        check_gpu()

    if transformers_installed:
        check_transformers()
        check_model("roberta-base")
    else:
        install_library("transformers")
        check_transformers()
        check_model("roberta-base")

    print("\nSetup Check Complete!")
