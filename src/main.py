from flask import Flask, request, render_template, redirect, url_for, session,flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from preprocessing import preprocess_text
from similarity import calculate_similarity
from detection import detect_plagiarism
from database import add_to_database as sqlalchemy_add_to_database
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
from sklearn.feature_extraction.text import TfidfVectorizer



matplotlib.use('Agg')  # Use 'Agg' for non-interactive backend suitable for web servers

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# SQLAlchemy configurations
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:9900@localhost/plagiarism_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'muzz28'  

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Define the User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Define the Document model
class Document(db.Model):
    __tablename__ = 'documents'
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        user = User.query.filter_by(username=username).first()
        if user:
            return 'user name already exist'
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user:
            # User exists, check password
            if check_password_hash(user.password, password):
                login_user(user)
                return redirect(url_for('home'))
            else:
                return 'wrong password'
        else:
            return render_template('login.html', error="No user found")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    return response

@app.route('/db_detection', methods=['GET', 'POST'])
@login_required
def db_detection():
    if request.method == 'POST':
        user_text = request.form['text']
        db_texts = [doc.text for doc in Document.query.all()]

        # Preprocess the user text and database texts
        preprocessed_user_text = preprocess_text(user_text)
        preprocessed_db_texts = [preprocess_text(text) for text in db_texts]

        # Initialize the vectorizer and fit_transform with preprocessed texts
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([preprocessed_user_text] + preprocessed_db_texts)

        # Extract feature names
        feature_names = vectorizer.get_feature_names_out()

        # Extract TF-IDF scores for user text
        tfidf_scores_user = tfidf_matrix[0].toarray()[0]

        # Extract TF-IDF scores for database texts
        tfidf_scores_db = tfidf_matrix[1:].toarray()

        # Create dictionaries of terms with their scores
        terms_user = {term: score for term, score in zip(feature_names, tfidf_scores_user) if score > 0}

        # Create dictionary of common terms with average scores
        common_terms_scores = {}
        for doc_scores in tfidf_scores_db:
            terms_db = {term: score for term, score in zip(feature_names, doc_scores) if score > 0}
            common_terms = set(terms_user.keys()) & set(terms_db.keys())
            for term in common_terms:
                if term not in common_terms_scores:
                    common_terms_scores[term] = []
                common_terms_scores[term].append(terms_db[term])

        # Find top three common terms based on their average score
        common_terms_avg_scores = {
            term: sum(scores) / len(scores)
            for term, scores in common_terms_scores.items()
        }
        sorted_common_terms = sorted(common_terms_avg_scores.items(), key=lambda x: x[1], reverse=True)[:3]

        # Prepare top common terms for rendering
        top_common_terms = {term: avg_score for term, avg_score in sorted_common_terms}

        # Perform plagiarism detection
        results, similarity_scores = detect_plagiarism(user_text, db_texts)
        
        # Plot similarity scores
        plt.figure(figsize=(8, 5))
        plt.bar(range(len(similarity_scores)), similarity_scores, color='yellow')
        plt.xlabel('Document Index')
        plt.ylabel('Similarity Score')
        plt.title('Similarity Scores with Database Documents')
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        # Render the result with top common terms
        return render_template('result.html', results=results, detection_type='database', plot_url=plot_url, top_common_terms=top_common_terms)

    return render_template('db_detection.html')


@app.route('/add_to_db', methods=['POST'])
@login_required
def add_to_db():
    user_text = request.form['text']
    sqlalchemy_add_to_database(user_text)
    return render_template('index.html', message="Content added to the database successfully.")

@app.route('/direct_comparison', methods=['GET', 'POST'])
def direct_comparison():
    if request.method == 'POST':
        text1 = request.form['text1']
        text2 = request.form['text2']
        processed_text1 = preprocess_text(text1)
        processed_text2 = preprocess_text(text2)

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([processed_text1, processed_text2])
        feature_names = vectorizer.get_feature_names_out()

        # Extract TF-IDF scores
        tfidf_scores1 = tfidf_matrix[0].toarray()[0]
        tfidf_scores2 = tfidf_matrix[1].toarray()[0]

        # Create dictionaries of terms with their scores
        terms1 = {term: score for term, score in zip(feature_names, tfidf_scores1) if score > 0}
        terms2 = {term: score for term, score in zip(feature_names, tfidf_scores2) if score > 0}

        # Find common terms
        common_terms = set(terms1.keys()) & set(terms2.keys())
        
        # Extract top three common terms based on their average scores
        common_terms_scores = {term: (terms1[term], terms2[term]) for term in common_terms}
        sorted_common_terms = sorted(common_terms_scores.items(), key=lambda x: (x[1][0] + x[1][1]) / 2, reverse=True)[:3]

        # Prepare top common terms for rendering
        top_common_terms = {term: scores for term, scores in sorted_common_terms}


        
        similarity_score = calculate_similarity(tfidf_matrix[0].toarray()[0], tfidf_matrix[1].toarray()[0]) * 100
        
        labels = ['Similarity', 'Difference']
        sizes = [similarity_score, 100 - similarity_score]
        colors = ['#66b3ff','#f66fdd']
        plt.figure(figsize=(6, 4))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return render_template('result.html', similarity_score=similarity_score, detection_type='direct', plot_url=plot_url, top_common_terms=top_common_terms)
    return render_template('direct_comparison.html')
if __name__ == '__main__':
    app.run(debug=True)
