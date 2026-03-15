from flask import Flask, request, render_template, session
import os
import sqlite3
import pickle
import re
from werkzeug.utils import secure_filename
from train_model import train_model_and_plot
import torch
torch.set_grad_enabled(False)
import transformers
transformers.torch = torch
from transformers import pipeline
from flask import send_from_directory, abort
from flask import Response
import datetime
from flask import redirect

app = Flask(__name__)
app.secret_key = "123"

# ---------------- BASIC ROUTES ---------------- #

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/admin")
def admin():
    return render_template("Admin/Admin.html")

@app.route("/AdminLogAction", methods=["POST"])
def AdminAction():
    username = request.form["uname"]
    password = request.form["password"]

    if username == "Admin" and password == "Admin":
        return render_template("Admin/AdminHome.html")
    else:
        return render_template("Admin/Admin.html", msg="Login Failed..!!")

@app.route("/AdminHome")
def AdminHome():
    return render_template("Admin/AdminHome.html")



ALLOWED_EXTENSIONS = {"txt"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/UploadAction", methods=["POST", "GET"])
def UploadAction():

    base_dir = "New_Dataset"
    full_text_dir = os.path.join(base_dir, "full_text")
    summary_dir = os.path.join(base_dir, "summary")

    full_text_files = []
    summary_files = []

    if os.path.exists(full_text_dir):
        full_text_files = [
            f for f in os.listdir(full_text_dir)
            if f.endswith(".txt")
        ]

    if os.path.exists(summary_dir):
        summary_files = [
            f for f in os.listdir(summary_dir)
            if f.endswith(".txt")
        ]

    total_files = len(full_text_files) + len(summary_files)

    return render_template(
        "Admin/UploadResult.html",
        total_files=total_files,
        full_text_count=len(full_text_files),
        summary_count=len(summary_files),
        full_text_files=full_text_files,
        summary_files=summary_files
    )

@app.route("/train_model")
def train_model():
    if os.path.exists("models/model.pkl"):
        return render_template(
            "Admin/train_result.html",
            msg="Model Already Trained",
            trained=True
        )
    else:
        results, best_acc, best_model_name = train_model_and_plot()
        return render_template(
            "Admin/train_result.html",
            results=results,
            best_acc=best_acc,
            best_model_name=best_model_name
        )

@app.route("/graph")
def graph():
    return render_template('Admin/Graph.html')
# ---------------- USER AUTH ---------------- #

@app.route("/register")
def register():
    return render_template("Client/Register.html")

@app.route("/RegAction", methods=["POST"])
def RegAction():
    con = sqlite3.connect("Legal.db")
    cur = con.cursor()

    cur.execute(
        "INSERT INTO user VALUES (null,?,?,?,?,?,?)",
        (
            request.form["name"],
            request.form["email"],
            request.form["mobile"],
            request.form["address"],
            request.form["uname"],
            request.form["password"],
        ),
    )

    con.commit()
    con.close()
    return render_template("Client/Register.html", msg="Registration Successful..!!")

@app.route("/client")
def client():
    return render_template("Client/Client.html")

@app.route("/LoginAction", methods=["POST"])
def LoginAction():
    con = sqlite3.connect("Legal.db")
    cur = con.cursor()

    cur.execute(
        "SELECT * FROM user WHERE username=? AND password=?",
        (request.form["username"], request.form["password"]),
    )
    data = cur.fetchone()

    if data is None:
        return render_template("Client/Client.html", msg="Login Failed..!!")
    else:
        session["name"] = data[1]
        session["email"] = data[2]
        return render_template("Client/ClientHome.html", name=data[1], email=data[2])

@app.route("/ClientHome")
def ClientHome():
    return render_template("Client/ClientHome.html")

@app.route("/Upload")
def Upload():
    return render_template("Client/Upload.html")

# ---------------- NLP MODEL (SAFE) ---------------- #

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=-1
)

def summarize_text(text):
    summary = summarizer(
        text,
        max_length=130,
        min_length=30,
        do_sample=False,
        truncation=True
    )
    return summary[0]["summary_text"]

# ---------------- LEGAL LOGIC ---------------- #

def extract_key_points(text):
    points = []

    deadline = re.search(r"within\s+\d+\s+days", text, re.IGNORECASE)
    if deadline:
        points.append(f"Payment deadline: {deadline.group()}")

    penalty = re.search(r"penalty\s+of\s+INR\s+\d+.*?day", text, re.IGNORECASE)
    if penalty:
        points.append(f"Penalty: {penalty.group()}")

    if "terminate" in text.lower():
        points.append("Termination clause exists")

    return points

def interpret_prediction(pred):
    if pred == 0:
        return "You can proceed safely, but read the document carefully."
    elif pred == 1:
        return "Review payment and termination clauses carefully."
    else:
        return "Pay before the deadline to avoid penalties."

# ---------------- FILE UPLOAD + OUTPUT ---------------- #

def save_history(username, email, filename, summary, key_points, advice, prediction):
    con = sqlite3.connect("Legal.db")
    cur = con.cursor()

    # Insert record
    cur.execute("""
        INSERT INTO history
        (username, email, filename, summary, key_points, advice, prediction)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        username,
        email,
        filename,
        summary,
        key_points,
        advice,
        prediction
    ))

    con.commit()
    con.close()



UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/TextAction", methods=["POST"])
def TextAction():
    uploaded_file = request.files["file"]

    if uploaded_file and uploaded_file.filename.endswith(".txt"):
        filename = secure_filename(uploaded_file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        uploaded_file.save(filepath)

        # Read uploaded text
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # NLP Processing
        summary = summarize_text(text)
        key_points = extract_key_points(text)

        # Load ML model
        model = pickle.load(open("models/model.pkl", "rb"))
        vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

        X = vectorizer.transform([summary])
        prediction = int(model.predict(X)[0])
        advice = interpret_prediction(prediction)

        # 🔹 SAVE HISTORY TO DATABASE
        save_history(
            session.get("name"),
            session.get("email"),
            filename,
            summary,
            "\n".join(key_points) if isinstance(key_points, list) else key_points,
            advice,
            prediction
        )

        return render_template(
            "Client/Summarized.html",
            summary=summary,
            key_points=key_points,
            advice=advice
        )

    return render_template("Client/Upload.html", msg="Invalid file format")

@app.route("/History")
def History():
    # User must be logged in
    if "email" not in session:
        return redirect("/client")

    con = sqlite3.connect("Legal.db")
    cur = con.cursor()

    # Fetch history for logged-in user only
    cur.execute("""
        SELECT id, filename, summary, key_points, advice, created_at
        FROM history
        WHERE email = ?
        ORDER BY created_at DESC
    """, (session["email"],))

    records = cur.fetchall()
    con.close()

    return render_template(
        "Client/History.html",
        records=records
    )

@app.route("/delete_history/<int:id>")
def delete_history(id):
    con = sqlite3.connect("Legal.db")
    cur = con.cursor()

    # Optional: get filename to delete uploaded file
    cur.execute("SELECT filename FROM history WHERE id=?", (id,))
    row = cur.fetchone()

    if row:
        filename = row[0]
        file_path = os.path.join("uploads", filename)
        if os.path.exists(file_path):
            os.remove(file_path)

    # Delete DB record
    cur.execute("DELETE FROM history WHERE id=?", (id,))
    con.commit()
    con.close()

    return redirect("/History")

from flask import send_from_directory

@app.route("/download/<int:hid>")
def download(hid):
    con = sqlite3.connect("Legal.db")
    cur = con.cursor()

    cur.execute("""
        SELECT filename, summary, key_points, advice, created_at
        FROM history
        WHERE id=?
    """, (hid,))
    row = cur.fetchone()
    con.close()

    if not row:
        return "Record not found", 404

    filename, summary, key_points, advice, created_at = row

    # -------- CREATE REPORT CONTENT --------
    report = f"""
AI-BASED LEGAL DOCUMENT ANALYSIS REPORT
=====================================

Original File Name:
{filename}

Analysis Date:
{created_at}

-------------------------------------
SUMMARY
-------------------------------------
{summary}

-------------------------------------
KEY POINTS
-------------------------------------
{key_points}

-------------------------------------
LEGAL ADVICE
-------------------------------------
{advice}

-------------------------------------
Generated By:
AI-Based Legal Document Summarizer & Law Advisor
"""

    # -------- DOWNLOAD RESPONSE --------
    download_name = f"AI_Legal_Report_{hid}.txt"

    return Response(
        report,
        mimetype="text/plain",
        headers={
            "Content-Disposition": f"attachment;filename={download_name}"
        }
    )
# ---------------- RUN ---------------- #

if __name__ == "__main__":
    app.run(debug=True)
