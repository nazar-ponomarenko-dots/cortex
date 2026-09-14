import sys
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings('ignore', category=UserWarning)

FALLBACK_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] | capitalize + ': ' + message['content'] + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}Assistant: {% endif %}"
)


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else './smollm-cortex'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, dtype=torch.bfloat16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = FALLBACK_TEMPLATE
        print('Tokenizer has no chat template; using a plain "Role: text" fallback.')
    model.eval()
    print('Cortex model loaded. Type your prompt ("exit" to quit).')

    messages = [{'role': 'system', 'content': 'You are a helpful and concise assistant.'}]

    while True:
        user_prompt = input('\n> ')
        if user_prompt.lower() in ['exit', 'quit']:
            break
        messages.append({'role': 'user', 'content': user_prompt})
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors='pt').to(device)

        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(output_tokens[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        messages.append({'role': 'assistant', 'content': response})
        if response:
            print(f'\n< {response}')
        else:
            print('\n< (the model produced nothing here - it is still undertrained; try another prompt)')


if __name__ == '__main__':
    main()
