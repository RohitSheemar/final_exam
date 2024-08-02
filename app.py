from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

app = Flask(__name__)
app.secret_key = 'supersecretkey'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file:
                data = pd.read_csv(file)
                results = analyze_data(data)
                return render_template('results.html', results=results)
        else:
            # Handle manual data entry
            data_str = request.form['data']
            data = pd.read_csv(io.StringIO(data_str))
            results = analyze_data(data)
            return render_template('results.html', results=results)
    return render_template('index.html')

def analyze_data(data):
    # Example analysis: Classification with the Iris dataset
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    X = iris_df[iris.feature_names]
    y = iris_df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True)
    
    # Plot feature importances
    importances = model.feature_importances_
    plt.figure()
    plt.title('Feature Importances')
    plt.barh(iris.feature_names, importances)
    plt.xlabel('Importance')
    plt.tight_layout()
    
    # Save plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_str = base64.b64encode(img.getvalue()).decode('utf-8')
    
    return {
        'report': report,
        'plot': img_str
    }

@app.route('/upload')
def upload_file():
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
