import requests
import base64
from io import BytesIO
from PIL import Image

class Llama32Model:
    def __init__(self, ollama_url="http://localhost:11411"):
        self.ollama_url = ollama_url
        self.model_name = "Llama 3.2 via Ollama"

    def preprocess_image(self, image: Image.Image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def generate_answer(self, image, question):
        image_str = self.preprocess_image(image)
        prompt = f"Image (base64): {image_str}\nQuestion: {question}\nAnswer:"
        data = {"prompt": prompt}
        try:
            response = requests.post(self.ollama_url + "/generate", json=data)
            if response.status_code == 200:
                resp_data = response.json()
                answer = resp_data.get("done", "").strip()
                return answer
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Exception: {str(e)}"
