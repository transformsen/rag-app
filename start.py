print("hello world")

from transformers import AutoTokenizer, AutoModel

# Load the CodeBERT-base model and tokenizer
model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Example code to generate embeddings for
code_examples = [
    "def add_numbers(a, b):\n    return a + b",
    "class MyClass:\n    def __init__(self, x):\n        self.x = x\n    def get_x(self):\n        return self.x",
    "import numpy as np\nX = np.random.rand(10, 5)\ny = np.random.rand(10)"
]

# Generate embeddings for the code examples
embeddings = []
for code in code_examples:
    input_ids = tokenizer.encode(code, return_tensors="pt")
    output = model(input_ids)[0][:, 0, :]  # Extract the CLS token embedding
    embeddings.append(output.detach().numpy())

print(f"Embeddings shape: {len(embeddings)}, {embeddings[0].shape}")