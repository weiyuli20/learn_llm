import os
import matplotlib.pyplot as plt
import urllib.request as request
from gpt2 import GPTModel,create_dataloader_v1,generate_text_simple
import tiktoken
import torch

def text_to_token_ids(text,tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids,tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch,target_batch = input_batch.to(device),target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1),target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model,tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model, encoded, max_new_tokens = 50, context_size = context_size)
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " ")) #compact print
    model.train()

def generate(model,idx,max_new_tokens,context_size,temperature=0.0,top_k = None, eos_id  =None):
    '''' topk 和温度采样'''
    for _ in range(max_new_tokens):
        idx_cond = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:,-1,:]

        if top_k is not None:
            top_logits,_ = torch.topk(logits,top_k)
            min_val = top_logits[:,-1]
            logits = torch.where(logits < min_val,torch.tensor(float("-inf")).to(device),logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits,dim=-1,keepdim=True)

        if idx_next == eos_id:
            break
        idx = torch.cat((idx,idx_next),dim=1)

    return idx
   
    model.eval()
    context_size = model.pos_emb.weight.shape[0]



def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()


def train_model_simple(model,train_loader,val_loader,optimizer,device, num_epoches,
                       eval_freq,eval_iter,start_context,tokenizer):
    #初始化
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1
    
    print(f"len_trainloader:{len(train_loader)}")
    #main train loop
    for epoch in range(num_epoches):
        model.train()

        for input_batch,target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch,target_batch,model,device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel() #numel()用来返回张量中元素的总数
            global_step +=1
 
            if global_step % eval_freq == 0:
                train_loss,val_loss = evaluate_model(model,train_loader,val_loader,device,eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
     
        #print a sample text after each epoch
        generate_and_print_sample(model,tokenizer,device,start_context)

    return train_losses, val_losses, track_tokens_seen


def main(gpt_config,settings):
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Download data if necessary

    file_path = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    if not os.path.exists(file_path):
        with request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text_data)
    else:
        with open(file_path,"r",encoding="utf-8") as f:
            text_data = f.read()

    #Initial model

    model = GPTModel(gpt_config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = settings["learning_rate"],
        weight_decay = settings["weight_decay"]
    )

    #set up dataloaders

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0)

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0)

    # Train model

    tokenizer = tiktoken.get_encoding("gpt2")

    train_losses,val_losses,tokens_seen = train_model_simple(model, train_loader, val_loader, optimizer, device,
                                                             num_epoches = settings["num_epochs"], 
                                                             eval_freq = 5,
                                                             eval_iter = 1, 
                                                             start_context ="Every effort moves you" , 
                                                             tokenizer = tokenizer)

    return train_losses,val_losses,tokens_seen,model


if __name__ == '__main__':
    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 256,  # Shortened context length (orig: 1024)
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-key-value bias
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 10,
        "batch_size": 2,
        "weight_decay": 0.1
    }

    train_losses, val_losses,tokens_seen,model = main(GPT_CONFIG_124M, OTHER_SETTINGS)
    
     # Plot results
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.jpg")

    # Save and load model
    torch.save(model.state_dict(), "model.pth")
    # model = GPTModel(GPT_CONFIG_124M)
    # model.load_state_dict(torch.load("model.pth", weights_only=True))

