import os
import glob
import json  # to create and store API call data
import numpy as np
import joblib

import torch  # deep learning library used for this project
import torch.nn as nn  # for the neural networks
import torch.nn.functional as F  # for the activation functions used ion models such as ReLU
import torch.optim as optim  # For the optimisation algorithms used in the model such as adam

from torch.utils.data import Dataset, DataLoader  # helps to handle the large batches of data and load them
from PIL import Image
from torchvision import models, \
    transforms  # used to help normalise byteplots for example, and models is used for the pre-trained models like resnet

from sklearn.feature_extraction.text import \
    CountVectorizer  # as explained in the paper the countvectoriser helps turn the api calls into the numerical vectors for the MLP
from sklearn.metrics import accuracy_score, recall_score, f1_score, \
    roc_auc_score  # used to calculate the evalaution metrics

# the directories of the dataset
BYTEPLOT_BASE = r"C:\Users\samue\Desktop\newDataset\byteplots"
BIGRAM_BASE = r"C:\Users\samue\Desktop\newDataset\bigrams"
APICALLS_BENIGN = r"C:\Users\samue\Desktop\newDataset\apicalls"
APICALLS_MALWARE = r"C:\Users\samue\Desktop\newDataset\apicalls_malware"

# the hyperparameters
EPOCHS_BYTEPLOT = 15
EPOCHS_API = 15
EPOCHS_BIGRAM = 15
EPOCHS_FUSION = 8
LR_BYTEPLOT = 5e-4
LR_API = 5e-4
LR_BIGRAM = 5e-4
LR_FUSION = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 16

# to make sure our pc used the GPU, which is much faster
device = "cuda" if torch.cuda.is_available() else "cpu"


# This section handles the data pre-processing and all the sampoling colelction before the model is trained.
# gets the actual filename of a byteplot, so its unique hash, which all byteplots and api calls etc have as their prefix.
def get_core_filename(byteplot_path):
    filename = os.path.basename(byteplot_path)
    return filename.replace("_byteplot.png", "")


# This part uses the obtained hash above and finds the corresponding api call .json file
def find_corresponding_api_path(core_filename, split, is_malware):
    api_filename = core_filename + "_apis.json"
    if is_malware:
        base_dir = APICALLS_MALWARE
    else:
        base_dir = APICALLS_BENIGN
    path = os.path.join(base_dir, split, api_filename)
    return path if os.path.exists(path) else None


# Iterates through and labels byteplots (1 for malware, 0 for benign)
def collect_samples_for_byteplot(split):
    samples = []

    mal_dir = os.path.join(BYTEPLOT_BASE, "malware", split)
    for fp in glob.glob(os.path.join(mal_dir, "*_byteplot.png")):
        samples.append((fp, 1))

    ben_dir = os.path.join(BYTEPLOT_BASE, "benign", split)
    for fp in glob.glob(os.path.join(ben_dir, "*_byteplot.png")):
        samples.append((fp, 0))
    return samples


# Iterates through and labels bigrams (1 for malware, 0 for benign)
def collect_samples_for_bigram(split):
    samples = []

    mal_dir = os.path.join(BIGRAM_BASE, "malware", split)
    for fp in glob.glob(os.path.join(mal_dir, "*.png")):
        samples.append((fp, 1))
    #
    ben_dir = os.path.join(BIGRAM_BASE, "benign", split)
    for fp in glob.glob(os.path.join(ben_dir, "*.png")):
        samples.append((fp, 0))
    return samples


# Iterates through and labels api calls (1 for malware, 0 for benign)
def collect_samples_for_api(split):
    samples = []

    mal_dir = os.path.join(BYTEPLOT_BASE, "malware", split)
    for bp in glob.glob(os.path.join(mal_dir, "*_byteplot.png")):
        core = get_core_filename(bp)
        api_path = find_corresponding_api_path(core, split, True)
        if api_path:
            samples.append((bp, api_path, 1))

    ben_dir = os.path.join(BYTEPLOT_BASE, "benign", split)
    for bp in glob.glob(os.path.join(ben_dir, "*_byteplot.png")):
        core = get_core_filename(bp)
        api_path = find_corresponding_api_path(core, split, False)
        if api_path:
            samples.append((bp, api_path, 0))

    return samples


class ByteplotDataset(Dataset):
    # in this class the byteplots are normalised to the correct input size for cnns then converted to a tensor for the model.
    def __init__(self, samples):
        self.samples = samples
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, lbl = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(lbl, dtype=torch.float32)


class BigramDataset(Dataset):
    # The same goes for the bigrams, they are normalised then converted to tensor.
    def __init__(self, samples):
        self.samples = samples
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, lbl = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(lbl, dtype=torch.float32)


class APIDataset(Dataset):
    # This section is jsut necessary to fill the img section of the tuple.
    # This is done so we had to only create one dataset loader tuple for each model, instead of one for each
    # so the cnn model just ignores the api calls (does nothing with them, but needs them), and the mlp ignores the images in the tuple
    def __init__(self, samples, vectorizer):
        self.samples = samples
        self.vectorizer = vectorizer
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        self.docs = []
        for (bp, api, lbl) in samples:
            if os.path.exists(api):
                with open(api, "r", encoding='utf-8') as f:
                    data = json.load(f)
                calls = data.get("api_calls", [])
                doc = " ".join(str(c) for c in calls)
                self.docs.append(doc)
            else:
                self.docs.append("")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        bp_path, ap_path, lbl = self.samples[idx]
        doc = self.docs[idx]
        img = Image.open(bp_path).convert("RGB")
        img = self.image_transform(img)

        if self.vectorizer:
            vec_sparse = self.vectorizer.transform([doc])
            vec_dense = vec_sparse.toarray()[0]
        else:
            vec_dense = np.zeros(1, dtype=np.float32)

        api_tensor = torch.tensor(vec_dense, dtype=torch.float32)
        return (img, api_tensor, torch.tensor(lbl, dtype=torch.float32))


# defining the byteplot model
class ByteplotResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        in_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_feats, 1)

    def forward(self, x):
        return self.resnet(x)


# defining the bigram model
class BigramResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        in_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_feats, 1)

    def forward(self, x):
        return self.resnet(x)


# defining the api mlp model
class APICallsMLP(nn.Module):
    # input_dim is the input of the vectorised api calls from inside the main
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# defining the byteplot modeldefining the fusion model
class FusionModel3(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# training loop that can be called to train any of the three models (not the fcnn),
# where model is the model to train and data loader is the dataset (or preproccessed data) that we have created
def train_epochs(model, loader, epochs=10, lr=5e-4, is_cnn=True):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    crit = nn.BCEWithLogitsLoss()

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            if is_cnn:  # if the model being trained is for byteplots or bigrams, do this
                img, lbl = batch
                img = img.to(device)
                lbl = lbl.to(device)
                optimizer.zero_grad()
                logits = model(img)
                loss = crit(logits.view(-1), lbl)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            else:  # otherwise train the mlp

                img, api_vec, lbl = batch
                img = img.to(device)
                api_vec = api_vec.to(device)
                lbl = lbl.to(device)
                optimizer.zero_grad()
                out = model(api_vec)
                loss = crit(out.view(-1), lbl)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Submodel epoch[{ep}/{epochs}] - loss={avg_loss:.4f}")
    return model  # returns the model


# collecting the output logits of the byteplot model for the fusion model, and retursn them with their labels
@torch.no_grad()
def collect_val_logits_byte(byte_model, loader):
    byte_model.eval()
    data = []
    for img, lbl in loader:
        img = img.to(device)
        logits = byte_model(img).view(-1)
        for i in range(len(lbl)):
            data.append((logits[i].item(), lbl[i].item()))
    return data


# Simirlarly collects the output logit of the bigram model for the fusion model and returns them with their labels
@torch.no_grad()
def collect_val_logits_bigram(big_model, loader):
    big_model.eval()
    data = []
    for img, lbl in loader:
        img = img.to(device)
        log = big_model(img).view(-1)
        for i in range(len(lbl)):
            data.append((log[i].item(), lbl[i].item()))
    return data


# and the same for the api model
@torch.no_grad()
def collect_val_logits_api(api_model, loader):
    api_model.eval()
    data = []
    for im, ap, lbl in loader:
        ap = ap.to(device)
        out = api_model(ap).view(-1)
        for i in range(len(lbl)):
            data.append((out[i].item(), lbl[i].item()))
    return data


# This is the fusion model, it bundles all the logits from the other models , checks they match and then converts them to a
# tensor so it can be fed into the model.
class Fusion3Dataset(Dataset):

    def __init__(self, byte_list, api_list, big_list):

        self.samples = []
        n = len(byte_list)
        for i in range(n):
            byte_log, lbl1 = byte_list[i]
            api_log, lbl2 = api_list[i]
            big_log, lbl3 = big_list[i]

            if not (lbl1 == lbl2 == lbl3):
                print(f"Labels do not match {i} => {lbl1}, {lbl2}, {lbl3}")
            self.samples.append((byte_log, api_log, big_log, lbl1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        b, a, g, lbl = self.samples[idx]
        return (torch.tensor([b, a, g], dtype=torch.float32), torch.tensor(lbl, dtype=torch.float32))


# this function is the training loop for the fusion model, which takes the tensor input of the three logits which was created above.
# this model is effectively learning how much to weight or trust each logit.
def train_fusion_3(fusion_model, data_loader, epochs=5, lr=1e-3):
    fusion_model = fusion_model.to(device)  # make sure we are using gpu because otherwise it will be too slow
    opt = optim.Adam(fusion_model.parameters(), lr=lr, weight_decay=1e-4)  # set up adam optimiser etc.
    crit = nn.BCEWithLogitsLoss()  # for binary classificaiton

    for ep in range(1, epochs + 1):
        fusion_model.train()
        total_loss = 0.0
        for triple_logits, label in data_loader:
            triple_logits = triple_logits.to(device)
            label = label.to(device)
            opt.zero_grad()
            out = fusion_model(triple_logits).view(-1)
            loss = crit(out, label)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        print(f"Fusion epoch[{ep}/{epochs}] - loss={avg_loss:.4f}")

        return fusion_model

# This is creating the "dataset" of the logits so they can be loaded in batches for the fcnn.
    class FusionValDS(Dataset):
        def __init__(self, bp_list, api_list, big_list):
            self.bp_list = bp_list
            self.api_list = api_list
            self.big_list = big_list

        def __len__(self):
            return len(self.bp_list)

        def __getitem__(self, idx):
            bp_log, lblA = self.bp_list[idx]
            api_log, lblB = self.api_list[idx]
            big_log, lblC = self.big_list[idx]
            # check labels match
            if not (lblA == lblB == lblC):
                print(f"Warning mismatched labels between: {lblA}, {lblB}, {lblC}")

            return torch.tensor([bp_log, api_log, big_log], dtype=torch.float32), torch.tensor(lblA,
                                                                                               dtype=torch.float32)


def main():
    # Checking that gpu is used before the model is run
    print("[INFO] Device:", device)
    # This collects all the sampels for the laoders at the start
    train_bp = collect_samples_for_byteplot("train")
    val_bp = collect_samples_for_byteplot("val")
    test_bp = collect_samples_for_byteplot("test")

    train_big = collect_samples_for_bigram("train")
    val_big = collect_samples_for_bigram("val")
    test_big = collect_samples_for_bigram("test")

    train_api = collect_samples_for_api("train")
    val_api = collect_samples_for_api("val")
    test_api = collect_samples_for_api("test")

    # makign sure that the correct sets have been collected for each part before the whole model is run.
    print(f"Train: byteplots={len(train_bp)}, bigrams={len(train_big)}, api={len(train_api)}")
    print(f"Val:   byteplots={len(val_bp)},   bigrams={len(val_big)},   api={len(val_api)}")
    print(f"Test:  byteplots={len(test_bp)},  bigrams={len(test_big)},  api={len(test_api)}")

    # this prepares the api calls for vectorisation by turning them into a string
    docs_for_train = []
    for (bp, ap, lbl) in train_api:
        with open(ap, "r", encoding='utf-8') as f:
            data = json.load(f)
        calls = data.get("api_calls", [])
        doc = " ".join(str(c) for c in calls)
        docs_for_train.append(doc)

    # This defines the count vectoriser for the api calls

    vectorizer = CountVectorizer(lowercase=False, stop_words=None, token_pattern=r"\S+")
    vectorizer.fit(docs_for_train)
    input_dim = len(vectorizer.vocabulary_)

    # This section takes all the bytepltos collected at the start of the script
    # from collect_samples_for_byteplot, and puts it inside an object, so they can
    # be turned into tenssors, normalised and loaded in batches etc.
    # the same goes for bigrams and api calls
    ds_bp_train = ByteplotDataset(train_bp)
    ds_bp_val = ByteplotDataset(val_bp)
    ds_bp_test = ByteplotDataset(test_bp)

    ds_big_train = BigramDataset(train_big)
    ds_big_val = BigramDataset(val_big)
    ds_big_test = BigramDataset(test_big)

    ds_api_train = APIDataset(train_api, vectorizer)
    ds_api_val = APIDataset(val_api, vectorizer)
    ds_api_test = APIDataset(test_api, vectorizer)

    # This is the dataloader that puts each part of each data set in batches
    ld_bp_train = DataLoader(ds_bp_train, batch_size=BATCH_SIZE, shuffle=True)
    ld_bp_val = DataLoader(ds_bp_val, batch_size=BATCH_SIZE, shuffle=False)
    ld_bp_test = DataLoader(ds_bp_test, batch_size=BATCH_SIZE, shuffle=False)

    ld_big_train = DataLoader(ds_big_train, batch_size=BATCH_SIZE, shuffle=True)
    ld_big_val = DataLoader(ds_big_val, batch_size=BATCH_SIZE, shuffle=False)
    ld_big_test = DataLoader(ds_big_test, batch_size=BATCH_SIZE, shuffle=False)

    ld_api_train = DataLoader(ds_api_train, batch_size=BATCH_SIZE, shuffle=True)
    ld_api_val = DataLoader(ds_api_val, batch_size=BATCH_SIZE, shuffle=False)
    ld_api_test = DataLoader(ds_api_test, batch_size=BATCH_SIZE, shuffle=False)

    # This part instantiates the models created earlier and calls the training loop which will inturn use each batch created above.
    bp_model = ByteplotResNet()
    big_model = BigramResNet()
    api_model = APICallsMLP(input_dim)

    print("\n---Train ByteplotResNet---")
    bp_model = train_epochs(bp_model, ld_bp_train, epochs=EPOCHS_BYTEPLOT, lr=LR_BYTEPLOT, is_cnn=True)

    print("\n---Train BigramResNet---")
    big_model = train_epochs(big_model, ld_big_train, epochs=EPOCHS_BIGRAM, lr=LR_BIGRAM, is_cnn=True)

    print("\n---Train APICallsMLP---")
    api_model = train_epochs(api_model, ld_api_train, epochs=EPOCHS_API, lr=LR_API, is_cnn=False)



    # This part now runs the valdiation sets through the training models (which is now in eval mode)
    # A logit and label output is provided for each, for the fcnn
    val_bp_list = []
    for img, lbl in ld_bp_val:  # the validation data loader is looped through
        img = img.to(device)  # each batch is sent to the gpu
        log = bp_model(img).view(-1)
        for i in range(len(lbl)):
            val_bp_list.append((log[i].item(), lbl[i].item()))

    val_big_list = []
    for img, lbl in ld_big_val:
        img = img.to(device)
        l = big_model(img).view(-1)
        for i in range(len(lbl)):
            val_big_list.append((l[i].item(), lbl[i].item()))

    val_api_list = []
    for im, ap, lbl in ld_api_val:
        ap = ap.to(device)
        out = api_model(ap).view(-1)
        for i in range(len(lbl)):
            val_api_list.append((out[i].item(), lbl[i].item()))

    if not (len(val_bp_list) == len(val_api_list) == len(val_big_list)):
        print("Validation sets do not match")
        return



    # Creating a dataset and loader for the validation logits
    fusion_val_ds = FusionValDS(val_bp_list, val_api_list, val_big_list)
    fusion_val_ld = DataLoader(fusion_val_ds, batch_size=16, shuffle=True)
    # setting parameters for fusion model
    fusion_model = FusionModel3()
    fusion_model = train_fusion_3(fusion_model, fusion_val_ld, epochs=EPOCHS_FUSION, lr=LR_FUSION)


    print("\n--- Evaluating the final pipeline on the test set ---")
    # sets models to evaluation after training, so that when we run the valdiation sets
    # through the training loops so it doesnt start adjusting dropout or another hyperparameter etc, as it is valdiating not training
    bp_model.eval()
    big_model.eval()
    api_model.eval()
    fusion_model.eval()

    all_preds = []
    all_labels = []

    # sending test data through dataloader and running each model to obtain the logits, preparing for fusion
    test_bp_list = []
    for img, lbl in ld_bp_test:
        img = img.to(device)
        log = bp_model(img).view(-1)
        for i in range(len(lbl)):
            test_bp_list.append((log[i].item(), lbl[i].item()))

    test_big_list = []
    for img, lbl in ld_big_test:
        img = img.to(device)
        l = big_model(img).view(-1)
        for i in range(len(lbl)):
            test_big_list.append((l[i].item(), lbl[i].item()))

    test_api_list = []
    for im, ap, lbl in ld_api_test:
        ap = ap.to(device)
        o = api_model(ap).view(-1)
        for i in range(len(lbl)):
            test_api_list.append((o[i].item(), lbl[i].item()))

    if not (len(test_bp_list) == len(test_big_list) == len(test_api_list)):
        print("The length of test sets are mismatched.")
        return

    #get byteplot model logit and label for test set, same for bigram model logit and api model logit.
    for i in range(len(test_bp_list)):
        bp_log, lbA = test_bp_list[i]
        bg_log, lbB = test_big_list[i]
        ap_log, lbC = test_api_list[i]
        if not (lbA == lbB == lbC):
            print(f"Label mismatch {lbA}, {lbB}, {lbC} at index {i}") #contniously check labels are matched as otherwise unrealistic results
        triple_in = torch.tensor([bp_log, ap_log, bg_log], dtype=torch.float32).to(device).unsqueeze(0)  # shape(1,3) is needed for the mlp (so squeeze does this), this line creates the tensor for the fusion model tho.
        out = fusion_model(triple_in).view(-1)
        pred = (torch.sigmoid(out) >= 0.5).long().cpu().item()  #use the sigmoid function to convert the logits into a probability.
        all_preds.append(pred) #this saves what the model predicted along with the true labels to calculate evaluation metrics.
        all_labels.append(lbA)

    all_preds = np.array(all_preds) #thsi converts the predictions and labels into NumPy arrays so that sklearn can work out the metrics.
    all_labels = np.array(all_labels).astype(int)

    test_acc = accuracy_score(all_labels, all_preds) #output scores!
    test_rec = recall_score(all_labels, all_preds, pos_label=1)
    test_f1 = f1_score(all_labels, all_preds, pos_label=1)

    print(f"[TEST] Accuracy={test_acc:.4f}, Recall={test_rec:.4f}, F1={test_f1:.4f}")


if __name__ == "__main__": #everythign needs to be wrapped in this otherwise there will be problems with memory management and the script wont run.
    main()
