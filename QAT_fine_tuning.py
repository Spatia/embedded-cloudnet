import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import Unet_1M, Unet_1M_Q
from cloud_dataset import CloudDataset
import unet_parts
import torch.ao.quantization as quantization

if __name__ == "__main__":
    LEARNING_RATE = 5e-6
    BATCH_SIZE = 12
    EPOCHS = 3
    DATA_PATH = "./dataset"
    ORIGINAL_MODEL_SAVE_PATH = "./models/unet_1M.pth"
    MODEL_SAVE_PATH = "unet_1M_QAT_FT_int8.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = CloudDataset(DATA_PATH)

    generator = torch.Generator().manual_seed(42)
    fine_tuning_dataset, _ = random_split(train_dataset, [0.2, 0.8], generator=generator)

    train_dataloader = DataLoader(dataset=fine_tuning_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    print(f"Loading model from Unet_1M for PTQ INT8 Quantization...")
    
    # 1. Charger les poids originaux du modèle Unet_1M
    original_state_dict = torch.load(ORIGINAL_MODEL_SAVE_PATH, map_location="cpu")
    
    # 2. Convertir les clés du dictionnaire d'états
    new_state_dict = {}
    for key, value in original_state_dict.items():
        new_key = key
        # Remplacement pour DoubleConv -> DoubleConv_Q
        if 'conv_op.0.' in key:
            new_key = key.replace('conv_op.0.', 'conv1.')
        elif 'conv_op.1.' in key:
            new_key = key.replace('conv_op.1.', 'bn1.')
        elif 'conv_op.3.' in key:
            new_key = key.replace('conv_op.3.', 'conv2.')
        elif 'conv_op.4.' in key:
            new_key = key.replace('conv_op.4.', 'bn2.')
            
        new_state_dict[new_key] = value

    # 3. Instancier le modèle avec l'architecture intégrant les QuantStubs et charger les nouveaux poids
    model = Unet_1M_Q(in_channels=4, num_classes=1)
    
    # strict=False permet d'ignorer les avertissements liés aux quant/dequant 
    # qui n'ont pas de poids à charger
    model.load_state_dict(new_state_dict, strict=False)

    model.eval()
    
    # Fusion des couches
    for module in model.modules():
        if isinstance(module, unet_parts.DoubleConv_Q):
            module.fuse_model()
            
    # Configuration QAT
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    
    per_tensor_qconfig = torch.ao.quantization.QConfig(
        activation=model.qconfig.activation,
        weight=torch.ao.quantization.default_weight_fake_quant
    )
    for module in model.modules():
        if isinstance(module, nn.ConvTranspose2d):
            module.qconfig = per_tensor_qconfig

    model.train()
            
    quantized_model = quantization.prepare_qat(model)

    # Transfert sur GPU pour l'entraînement
    quantized_model.to(device)

    optimizer = optim.AdamW(quantized_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # 5. Boucle d'entraînement
    for epoch in tqdm(range(EPOCHS)):
        quantized_model.train()
        train_running_loss = 0
        
        for idx, (img, mask) in enumerate(tqdm(train_dataloader)):
            img = img.float().to(device)
            mask = mask.float().to(device)

            y_pred = quantized_model(img)
            optimizer.zero_grad()

            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(quantized_model.parameters(), max_norm=1.0)
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)

        print("-"*30)
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print("-"*30)

    quantized_model.eval()
    quantized_model.to('cpu') 
    
    # On convertit le modèle entraîné (convertit les FakeQuantize en vraies couches INT8)
    final_int8_model = quantization.convert(quantized_model)
    
    torch.save(final_int8_model.state_dict(), MODEL_SAVE_PATH)