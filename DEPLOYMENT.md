# GitHub and Streamlit Community Cloud deployment

## 1. Add a trained model

The application needs these two files before it can make predictions:

```text
artifacts/models/mobilenetv2.keras
artifacts/models/mobilenetv2_labels.json
```

Generate them locally by placing your dataset in `data/garbage/` and running:

```powershell
.\.venv\Scripts\Activate.ps1
python train.py --data-dir data/garbage --model mobilenetv2 --epochs 15
```

If the model is larger than GitHub's 100 MB limit, store it with Git LFS or in a supported model store, then adjust `app.py` to download it at startup. Do not add datasets or checkpoints directly to ordinary Git commits.

## 2. Publish to GitHub

Create an empty repository named `recyclevision-garbage-classification`, then run:

```powershell
git remote add origin https://github.com/YOUR-USERNAME/recyclevision-garbage-classification.git
git branch -M main
git push -u origin main
```

## 3. Deploy on Streamlit Community Cloud

1. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Choose **Create app**, select the GitHub repository and branch `main`.
3. Set the entry point to `app.py`, then choose **Deploy**.
4. Confirm the app loads and upload an image.

`runtime.txt` selects Python 3.12. Streamlit installs packages from `requirements.txt` automatically.
