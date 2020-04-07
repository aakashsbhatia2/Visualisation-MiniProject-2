from flask import Flask, render_template, request, redirect, Response, jsonify
import pandas as pd
import json
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS

app = Flask(__name__)

@app.route("/", methods = ['GET', 'POST'])
def index():
    return render_template("index.html", data = "")

@app.route("/elbow-k-means", methods = ['POST'])
def get_elbow():
    global elbow_vals
    return jsonify(elbow_vals)

@app.route("/screeplot_full_data", methods = ['POST'])
def scree_plot_full_data():
    global df1
    data, list_intrinsic_dimensions, count_2 = calculate_intrinsic_dimensionality(df1)
    return jsonify(data)

@app.route("/screeplot_random_sample_data", methods = ['POST'])
def scree_plot_random_data():
    global df2
    data, list_intrinsic_dimensions, count_2 = calculate_intrinsic_dimensionality(df2)
    return jsonify(data)

@app.route("/screeplot_stratified_sample_data", methods = ['POST'])
def scree_plot_stratified_data():
    global df3
    data, list_intrinsic_dimensions, count_2 = calculate_intrinsic_dimensionality(df3)
    return jsonify(data)


@app.route("/scatter_plot_full_data_pc_values", methods = ['POST'])
def scatter_plot_full_data_pc_values():
    global df1
    data = calculate_top_PC_Vals(df1)
    return jsonify(data)

@app.route("/scatter_plot_random_sampled_data_pc_values", methods = ['POST'])
def scatter_plot_random_sampled_data_pc_values():
    global df2
    data = calculate_top_PC_Vals(df2)
    return jsonify(data)

@app.route("/scatter_plot_stratified_sampled_data_pc_values", methods = ['POST'])
def scatter_plot_stratified_sampled_data_pc_values():
    global df3
    data = calculate_top_PC_Vals(df3)
    return jsonify(data)

@app.route("/scatter_matrix_full_data", methods = ['POST'])
def scatter_matrix_full_data():
    global df1
    data_temp, list_intrinsic_dimensions, count_2 = calculate_intrinsic_dimensionality(df1)
    data = calculate_top_attributes(df1, list_intrinsic_dimensions,count_2)
    return jsonify(data)

@app.route("/scatter_matrix_random_sampled_data", methods = ['POST'])
def scatter_matrix_random_sampled_data():
    global df2
    data_temp, list_intrinsic_dimensions, count_2 = calculate_intrinsic_dimensionality(df2)
    data = calculate_top_attributes(df2, list_intrinsic_dimensions,count_2)
    return jsonify(data)

@app.route("/scatter_matrix_stratified_sampled_data", methods = ['POST'])
def scatter_matrix_stratified_sampled_data():
    global df3
    data_temp, list_intrinsic_dimensions, count_2 = calculate_intrinsic_dimensionality(df1)
    data = calculate_top_attributes(df3, list_intrinsic_dimensions, count_2)
    return jsonify(data)

@app.route("/mds_euclidean_full_data", methods = ['POST'])
def mds_euclidean_full_data():
    global df1
    data = calculate_mds(df1, 'euclidean')
    return jsonify(data)

@app.route("/mds_euclidean_random_data", methods = ['POST'])
def mds_euclidean_random_data():
    global df2
    data = calculate_mds(df2, 'euclidean')
    return jsonify(data)

@app.route("/mds_euclidean_stratified_data", methods = ['POST'])
def mds_euclidean_stratified_data():
    global df3
    data = calculate_mds(df3, 'euclidean')
    return jsonify(data)

@app.route("/mds_correlation_full_data", methods = ['POST'])
def mds_correlation_full_data():
    global df1
    data = calculate_mds(df1, 'correlation')
    return jsonify(data)

@app.route("/mds_correlation_random_data", methods = ['POST'])
def mds_correlation_random_data():
    global df2
    data = calculate_mds(df2, 'correlation')
    return jsonify(data)

@app.route("/mds_correlation_stratified_data", methods = ['POST'])
def mds_correlation_stratified_data():
    global df3
    data = calculate_mds(df3, 'correlation')
    return jsonify(data)


def calculate_intrinsic_dimensionality(df):
    count = 0
    count_2 = 0
    sum = 0
    sum_2 = 0
    ptg_data = []
    list_intrinsic_dimensions = []

    pca = PCA(n_components=13)

    x = StandardScaler().fit_transform(df.loc[:, :'Overall_Ranking'])
    principal_components = pca.fit_transform(x)
    data = pca.explained_variance_ratio_
    data = data.tolist()
    for i in data:
        sum += i
        if sum<0.80:
            count_2 +=1
            list_intrinsic_dimensions.append("PC" + str(count_2))

        ptg_data.append(sum)

    list_pca = []
    for i in data:
        count += 1
        list_pca.append("PC" + str(count))

    flag = 0
    list_pc_vector = []
    for i in range(len(data)):
        sum_2 += data[i]
        if sum_2 < 0.80:
            list_pc_vector.append("True")
        else:
            list_pc_vector.append("False")


    for i in range(len(list_pc_vector)):
        if list_pc_vector[i+1] == 'False':
            for j in range(i):
                list_pc_vector[j] = 'False'
            break

    mapping = {"PC": list_pca, "Value": data, "Percent_data": ptg_data, "PC_T_F": list_pc_vector}
    df_combined = pd.DataFrame(mapping)
    mapping = df_combined.to_dict(orient="records")
    chart_data = json.dumps(mapping, indent=2)
    data = {'chart_data': chart_data}
    return data, list_intrinsic_dimensions, count_2

def calculate_top_PC_Vals(df):
    pca = PCA(n_components=2)

    x = MinMaxScaler().fit_transform(df.loc[:, :'Overall_Ranking'])
    pc = pca.fit_transform(x)
    data = pca.explained_variance_ratio_
    principal_Df = pd.DataFrame(data=pc, columns=['PC1', 'PC2'])
    principal_Df['University'] = df['University']
    principal_Df['Country'] = df['Country']
    principal_Df['Rank'] = df['Rank']

    mapping = principal_Df.to_dict(orient="records")
    chart_data = json.dumps(mapping, indent=2)
    data = {'chart_data': chart_data}
    return data

def calculate_top_attributes(df, list_intrinsic_dimensions, count):
    df_temp = df.loc[:, :'Overall_Ranking']
    list_country = df['Country'].tolist()
    list_uni = df['University'].tolist()
    list_rank = df['Rank'].tolist()
    pca = PCA(n_components=count)

    x = StandardScaler().fit_transform(df_temp)
    pc1 = pca.fit_transform(x)
    data = pca.explained_variance_ratio_
    data = data.tolist()
    loadings = pd.DataFrame(pca.components_.T, columns= list_intrinsic_dimensions,
                            index=df_temp.columns)

    for i in loadings:
        loadings[i] = loadings[i] ** 2
    loadings['sum'] = loadings.sum(axis=1)
    df_final = df_temp.loc[:, loadings.nlargest(3, ['sum']).index.tolist()]
    df_final['Country'] = list_country
    df_final['University'] = list_uni
    df_final['Rank'] = list_rank

    mapping = df_final.to_dict(orient="records")
    chart_data = json.dumps(mapping, indent=2)
    data = {'chart_data': chart_data}
    return data

def calculate_mds(df, distance):
    df_temp = df.loc[:, :'Overall_Ranking']
    df_scaled = StandardScaler().fit_transform(df_temp)
    df_scaled_final = pd.DataFrame(df_scaled)
    matrix = metrics.pairwise.pairwise_distances(df_scaled_final, metric=distance)
    list_country = df['Country'].tolist()
    list_uni = df['University'].tolist()
    list_rank = df['Rank'].tolist()

    mds = MDS(n_components=2, dissimilarity='precomputed', n_jobs=-1)

    values = mds.fit_transform(matrix)

    df_final = pd.DataFrame(values, columns= ['MDS1', 'MDS2'])
    df_final['Country'] = list_country
    df_final['University'] = list_uni
    df_final['Rank'] = list_rank

    mapping = df_final.to_dict(orient="records")
    chart_data = json.dumps(mapping, indent=2)
    data = {'chart_data': chart_data}
    return data

def create_random_samples():
    global df1
    df_random = df1.sample(frac=0.25)
    df_random.to_csv("random_sample.csv", index=False)

def create_random_stratified_samples():

    global df1

    sum_of_squared_distances = []
    k_vals = []
    elbow_values = []
    for k in range(1,18):
        k_vals.append(k)
        k_means = KMeans(n_clusters=k)
        model = k_means.fit(df1.loc[:, :'Overall_Ranking'])
        sum_of_squared_distances.append(k_means.inertia_ * 10**-12)
    elbow_values.append(k_vals)
    elbow_values.append(sum_of_squared_distances)
    final_elbow_vals = pd.DataFrame(elbow_values)
    final_elbow_values = final_elbow_vals.T
    final_elbow_values.columns = ['K', 'Value']
    mapping = final_elbow_values.to_dict(orient="records")
    chart_data = json.dumps(mapping, indent=2)
    data = {'chart_data': chart_data}

    k_means = KMeans(n_clusters=4)
    model = k_means.fit(df1.loc[:, :'Overall_Ranking'])
    y_hat = k_means.predict(df1.loc[:, :'Overall_Ranking'])
    labels = k_means.labels_
    df1['Cluster'] = y_hat

    df_c0 = df1.loc[df1['Cluster'] == 0]
    df_c0_sample = df_c0.sample(frac=0.25)

    df_c1 = df1.loc[df1['Cluster'] == 1]
    df_c1_sample = df_c1.sample(frac=0.25)

    df_c2 = df1.loc[df1['Cluster'] == 2]
    df_c2_sample = df_c2.sample(frac=0.25)

    df_c3 = df1.loc[df1['Cluster'] == 3]
    df_c3_sample = df_c3.sample(frac=0.25)

    df_strat_samples = df_c0_sample.append(df_c1_sample.append(df_c2_sample.append(df_c3_sample)))
    df_strat_final = df_strat_samples.drop(columns='Cluster')

    df_strat_final.to_csv("random_stratified.csv", index=False)
    return data

if __name__ == "__main__":
    df1 = pd.read_csv("data_full.csv")
    create_random_samples()
    df2 = pd.read_csv("random_sample.csv")
    elbow_vals = create_random_stratified_samples()
    df3 = pd.read_csv("random_stratified.csv")
    app.run(debug=True)
