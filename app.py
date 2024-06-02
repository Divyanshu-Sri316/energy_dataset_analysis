from flask import Flask, request, render_template
import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data

def calculate_statistics(data, column_name):
    mean_val = data[column_name].mean()
    median_val = data[column_name].median()
    mode_val = data[column_name].mode()[0]
    std_dev_val = data[column_name].std()
    corr_matrix = data.corr()

    statistics = {
        'mean': mean_val,
        'median': median_val,
        'mode': mode_val,
        'std_dev': std_dev_val,
        'corr_matrix': corr_matrix
    }
    return statistics

def generate_histogram(data, column_name, output_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column_name], bins=30, kde=True)
    plt.title(f'Histogram of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.savefig(output_path)
    plt.close()

def generate_scatter_plot(data, column_x, column_y, output_path):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=column_x, y=column_y, data=data)
    plt.title('Scatter Plot')
    plt.xlabel(column_x)
    plt.ylabel(column_y)
    plt.savefig(output_path)
    plt.close()

def generate_line_plot(data, column_x, column_y, output_path):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=column_x, y=column_y, data=data)
    plt.title('Line Plot')
    plt.xlabel(column_x)
    plt.ylabel(column_y)
    plt.savefig(output_path)
    plt.close()

def get_insights_from_llm(prompt, model_name='llama-2'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=500)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        data = read_csv(file_path)
        column_name = request.form['column_name']
        stats = calculate_statistics(data, column_name)

        generate_histogram(data, column_name, 'static/histogram.png')
        generate_scatter_plot(data, column_name, column_name, 'static/scatter_plot.png')
        generate_line_plot(data, column_name, column_name, 'static/line_plot.png')

        prompt = f"""
        The dataset has the following statistical parameters:
        - Mean of {column_name}: {stats['mean']}
        - Median of {column_name}: {stats['median']}
        - Mode of {column_name}: {stats['mode']}
        - Standard Deviation of {column_name}: {stats['std_dev']}

        The correlation matrix is as follows:
        {stats['corr_matrix']}

        Generate a comprehensive analysis and insights based on these statistics.
        """

        insights = get_insights_from_llm(prompt)

        return render_template('results.html', stats=stats, insights=insights)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
