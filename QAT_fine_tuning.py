import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import Unet, Unet_1M_Q
from cloud_dataset import CloudDataset
import unet_parts
import torch.ao.quantization as quantization

if __name__ == "__main__":
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 12
    EPOCHS = 20
    DATA_PATH = "./dataset"
    ORIGINAL_MODEL_SAVE_PATH = "./models/unet_1M.pth"
    MODEL_SAVE_PATH = "unet_QAT_FT.pth"
    QUANTIZATION = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = CloudDataset(DATA_PATH)

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    model = Unet_1M_Q(in_channels=4, num_classes=1).to(device)

    if QUANTIZATION:
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
        
        # ConvTranspose2d issue
        per_tensor_qconfig = torch.ao.quantization.QConfig(
            activation=model.qconfig.activation,
            weight=torch.ao.quantization.default_weight_fake_quant
        )
        for module in model.modules():
            if isinstance(module, nn.ConvTranspose2d):
                module.qconfig = per_tensor_qconfig
        
        model.train() 
        for module in model.modules():
            if isinstance(module, unet_parts.DoubleConv_Q):
                module.fuse_model()
                
        model = quantization.prepare_qat(model)

    #Print the number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(EPOCHS)):
        model.train()

        train_running_loss = 0
        for idx, (img,mask) in enumerate(tqdm(train_dataloader)):
            img = img.float().to(device)
            mask = mask.float().to(device)

            y_pred = model(img)
            optimizer.zero_grad()

            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)

        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)
                
                y_pred = model(img)
                loss = criterion(y_pred, mask)

                val_running_loss += loss.item()

            val_loss = val_running_loss / (idx + 1)

        torch.cuda.empty_cache()
        print("-"*30)
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print("-"*30)

    model.eval()
    model.to('cpu') # La conversion et l'inférence quantifiée INT8 sont généralement faites sur CPU
    quantized_model = quantization.convert(model)
    torch.save(quantized_model.state_dict(), MODEL_SAVE_PATH)