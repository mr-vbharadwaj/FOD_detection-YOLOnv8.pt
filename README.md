
# 🛫 FOD Detection – User Run Guide

This guide explains **how to run the application**, from setup to downloading the final output video.


## What This Application Does

- Takes a **video input** (runway footage)
- Detects **Foreign Object Debris (FOD)**
- Tracks objects to **avoid double counting**
- Generates an **annotated output video**
- Lets you **download the result**

The system works **out of the box** using a **pre-trained model** included in this repository.

**You do NOT need to train anything again.**

---

## System Requirements

- Windows 10 or 11  
- Python 3.9 – 3.11  
- Internet connection (first-time setup only)  
- No GPU required  

---

## Folder Overview (What You Need to Know)

You only need to care about **these files/folders**:

```
FOD/
├─ app.py            # The application you will run
├─ runs/             # Contains the pre-trained model (DO NOT DELETE)
├─ test_files/       # Optional: place videos here
├─ README.md         # This file
```

You can ignore all notebooks and training files.

---

# HOW TO RUN THE APPLICATION (STEP-BY-STEP)

Follow **exactly in this order**.

---

## STEP 1 — Open Command Prompt

- Press **Win + R**
- Type `cmd`
- Press Enter

Navigate to the project folder:

```bat
cd path\to\FOD
```

---

## STEP 2 — Create a Virtual Environment (One Time Only)

```bat
python -m venv venv
```

Activate it:

```bat
venv\Scripts\activate
```

You should now see:

```
(venv)
```

---

## STEP 3 — Install Required Software

```bat
pip install ultralytics opencv-python streamlit
```

IMPORTANT: This step is required only once.

---

## STEP 4 — Start the Application

Run the command:

```bat
streamlit run app.py
```

If asked for email, just hit enter to leave it blank and proceed ahead.

Your browser will open automatically.

---

## STEP 5 — Use the Application

In the browser:

1. Click **Upload Video**
2. Select any `.mp4`, `.avi`, or `.mov` file
3. Click **Run FOD Detection**
4. Wait until processing completes

Processing time depends on video length and your CPU.

---

## STEP 6 — Download the Output Video

After processing:

- The annotated video will play in the browser
- Click **Download Output Video**
- Save the file to your system

You’re done.

---

## Optional: Using Your Own Videos

You may upload:
- runway inspection footage
- drone videos
- static camera recordings

No special formatting is required.

---

## Troubleshooting

- Ensure Python is installed correctly
- Ensure `(venv)` is visible in the terminal
- Do not delete the `runs/` folder
- Restart the app if something goes wrong

---


