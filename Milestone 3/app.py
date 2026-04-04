import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from torchvision import models, transforms
from PIL import Image
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pcb_secret_key_123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:root@localhost:5432/pcb_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- Folders Setup ---
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
TEMPLATE_FOLDER = 'static/Template_images' 

for f in [UPLOAD_FOLDER, RESULT_FOLDER, TEMPLATE_FOLDER]:
    if not os.path.exists(f): os.makedirs(f)

# --- AI Model Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Missing_hole', 'Mouse_bite', 'Normal','Open_circuit', 'Short', 'Spur', 'Spurious_copper']


val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))

MODEL_PATH = 'pcb_final.pth'
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"AI Model Loaded Successfully on {DEVICE}!")
else:
    print(f"ERROR: {MODEL_PATH} not found!")

# --- Database Models ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class ScanHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    filename = db.Column(db.String(100))
    result_label = db.Column(db.String(100))
    date_posted = db.Column(db.DateTime, default=datetime.utcnow)

# --- Login Manager ---
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))



def get_prediction(img_path):
    # 1. LOAD IMAGE
    user_img_raw = cv2.imread(img_path)
    if user_img_raw is None: 
        return "Load Error", None, [], {}

    # 2. LOAD TEMPLATE
    all_temps = [f for f in os.listdir(TEMPLATE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not all_temps: 
        return "No Templates", None, [], {}
    
    img_t_raw = cv2.imread(os.path.join(TEMPLATE_FOLDER, all_temps[0]))
    
    # 3. PRE-PROCESS & ALIGNMENT
    img_i_res = cv2.resize(user_img_raw, (img_t_raw.shape[1], img_t_raw.shape[0]))
    res_img = img_i_res.copy()
    
    # Gaussian Blur for Noise Reduction 
    g_t = cv2.GaussianBlur(cv2.cvtColor(img_t_raw, cv2.COLOR_BGR2GRAY), (5,5), 0)
    g_i = cv2.GaussianBlur(cv2.cvtColor(img_i_res, cv2.COLOR_BGR2GRAY), (5,5), 0)

    # 4. SUBTRACTION
    diff = cv2.absdiff(g_t, g_i)
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY) 
    
    # Morphological cleaning to remove tiny noise
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Data collection containers
    defect_details = []
    summary_count = {}
    found_labels = []

    # 5. CLASS NAMES
    CLASS_NAMES = ['Missing_hole', 'Mouse_bite', 'Normal', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']

    for cnt in contours:
        # Area filter 
        if cv2.contourArea(cnt) > 8: 
            x, y, w, h = cv2.boundingRect(cnt)
            
            # --- ROI EXTRACTION 
            img_h, img_w = img_i_res.shape[:2]
            y1, y2 = max(0, y-10), min(img_h, y+h+10)
            x1, x2 = max(0, x-10), min(img_w, x+w+10)
            
            patch = img_i_res[y1:y2, x1:x2]

            if patch.size > 0:
                # 6. AI PREDICTION
                patch_res = cv2.resize(patch, (128, 128))
                patch_rgb = cv2.cvtColor(patch_res, cv2.COLOR_BGR2RGB) 
                
                input_tensor = val_transform(Image.fromarray(patch_rgb)).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    prob = torch.nn.functional.softmax(output, dim=1)
                    conf, pred = torch.max(prob, 1)
                
                idx = pred.item()
                label = CLASS_NAMES[idx]
                confidence = round(conf.item() * 100, 1)

                # 7. DRAWING & LOGGING 
                if label.lower() != 'normal':
                    found_labels.append(label)
                    
                    # Drawing logic 
                    color = (0, 0, 255) 
                    cv2.rectangle(res_img, (x-5, y-5), (x+w+5, y+h+5), color, 2)
                    cv2.putText(res_img, f"{label} {confidence}%", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Table & Summary data
                    defect_details.append({
                        'name': label,
                        'x': x,
                        'y': y,
                        'conf': f"{confidence}%"
                    })
                    summary_count[label] = summary_count.get(label, 0) + 1

    # 8. FINAL STATUS & SAVING
    if found_labels:
       
        status = f"Defects Detected: {', '.join(sorted(list(set(found_labels))))}"
    else:
        status = "PCB Normal"

    result_filename = "res_" + os.path.basename(img_path)
    cv2.imwrite(os.path.join(RESULT_FOLDER, result_filename), res_img)
    
    return status, result_filename, defect_details, summary_count
# --- ROUTES ---
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already exists!')
            return redirect(url_for('signup'))
        new_user = User(username=username, password=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()
        flash('Registration Successful!')
        return redirect(url_for('login'))
    return render_template('login.html', signup_mode=True)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid Credentials!')
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/scan', methods=['GET', 'POST'])
@login_required
def scan():
    result = None
    image = None
    defects = []
    summary = {}

    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            # 1. Image Save
            raw_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
            path = os.path.join(UPLOAD_FOLDER, raw_filename)
            file.save(path)
            
            # 2. Prediction
            result, image, defects, summary = get_prediction(path)
            
       
            try:
                new_scan = ScanHistory(
                    filename=image,         
                    result_label=result,      
                    user_id=current_user.id
                )
                db.session.add(new_scan)
                db.session.commit()
                print("Database Success: Scan saved to history!")
            except Exception as e:
                db.session.rollback()
                print(f"Database Error: {str(e)}")
            
            return render_template('scan.html', 
                                   result=result, 
                                   image=image, 
                                   defects=defects, 
                                   summary=summary)

    return render_template('scan.html', result=result, image=image, defects=defects, summary=summary)

@app.route('/history')
@login_required
def history():
    
    scans = ScanHistory.query.filter_by(user_id=current_user.id).order_by(ScanHistory.date_posted.desc()).all()
    return render_template('history.html', scans=scans)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
