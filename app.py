import gradio as gr
from transformers import ViltProcessor, ViltForQuestionAnswering

# Load model and processor
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def answer_question(image, text):
    try:
        encoding = processor(image, text, return_tensors="pt")
        outputs = model(**encoding)
        idx = outputs.logits.argmax(-1).item()
        return model.config.id2label[idx]
    except Exception as e:
        return "Error processing the image or question."

# --------- UI LAYOUT ----------
with gr.Blocks(title="Visual Question Answering") as demo:
    gr.Markdown("## 🤖 Multimodal AI: Visual Question Answering")

    with gr.Row():
        # LEFT: Image
        image_input = gr.Image(type="pil", label="Upload Image")

        # RIGHT: Question + Answer
        with gr.Column():
            question_input = gr.Textbox(
                label="Ask a question about the image",
                placeholder="What is happening in the image?"
            )
            submit_btn = gr.Button("Submit")
            answer_output = gr.Textbox(label="Answer")

    submit_btn.click(
        fn=answer_question,
        inputs=[image_input, question_input],
        outputs=answer_output
    )

demo.launch(share=True)

