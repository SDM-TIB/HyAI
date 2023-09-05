import pandas as pd
import os, sys, glob
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import Utility
import matplotlib.pyplot as plt


def call_semEP(threshold, cls_addres, file_addres):
    DIR_SEM_EP = "semEP-node"
    current_path = os.path.dirname(os.path.realpath(__file__))
    th = "{:.4f}".format(float(threshold))

    commd = current_path + "/" + DIR_SEM_EP + " " + file_addres + "donor.txt "+file_addres+"matrix_donor.tsv " + str(th)
    print('commd: ' + commd)
    os.system(commd)
    pattern = 'donor'

    results_folder = glob.glob(current_path + "/" + pattern + "-*")
    # print(results_folder)
    onlyfiles = [os.path.join(results_folder[0], f) for f in listdir(results_folder[0]) if
                 isfile(join(results_folder[0], f))]
    count = 0
    for filename in onlyfiles:
        # print(filename)
        key = "cluster-" + str(count) + '.txt'
        copyfile(filename, cls_addres + 'clusters/' + key)
        count += 1
    # dicc_clusters = get_dicc_clusters(onlyfiles)

    for r, d, f in os.walk(results_folder[0]):
        for files in f:
            os.remove(os.path.join(r, files))
        os.removedirs(r)
    return len(onlyfiles)


def METIS_Undirected_MAX_based_similarity_graph(cosine_matrix, cls_address_metis):
    metislines = []
    nodes = {"name": [], "id": []}
    kv = 1
    edges = 0
    for i, row in cosine_matrix.iterrows():
        val = ""
        ix = 1
        ledges = 0
        found = False
        for k in row.keys():
            if i != k and row[k] > 0:
                val += str(ix) + " " + str(int(row[k] * 100000)) + " "
                # Only one edge is counted between two nodes, i.e., (u,v) and (v, u) edges are counted as one
                # Self links are also ignored, Notive ix>kv
                # if ix > kv:
                ledges += 1
                found = True
            ix += 1
        if found:
            # This node is connected
            metislines.append(val.strip())
            edges += ledges
            nodes["name"].append(i)
            nodes['id'].append(str(kv))
        else:
            # disconnected RDF-MTs are given 10^6 value as similarity value
            metislines.append(str(kv) + " 100000")
            edges += 1
            # ---------
            nodes["name"].append(i)
            nodes['id'].append(str(kv))
            print(i)
            print(str(kv))

        kv += 1
    nodes = pd.DataFrame(nodes)
    #print(edges)
    numedges = edges // 2
    # == Save filemetis.graph to execute METIS algorithm ==
    ff = open(cls_address_metis + 'metis.graph', 'w+')
    ff.write(str(cosine_matrix.shape[0]) + " " + str(numedges) + " 001\n")
    met = [m.strip() + "\n" for m in metislines]
    ff.writelines(met)
    ff.close()
    return nodes


def call_metis(num_cls, nodes, cls_address_metis):
    # !sudo docker run -it --rm -v /media/rivas/Data1/Data-mining/KCAP-I40KG-Embeddings/I40KG-Embeddings/result/TransD/metis:/data kemele/metis:5.1.0 gpmetis metis.graph 2
    current_path = os.path.dirname(os.path.realpath(__file__))
    EXE_METIS = "sudo docker run -it --rm -v "
    DIR_METIS = ":/data kemele/metis:5.1.0 gpmetis"
    cls_addres = cls_address_metis[:-1]
    commd = EXE_METIS + current_path + '/' + cls_addres + DIR_METIS + " metis.graph " + str(num_cls)
    print(commd)
    os.system(commd)
    parts = open(cls_address_metis + 'metis.graph.part.' + str(num_cls)).readlines()
    parts = [p.strip() for p in parts]
    # == Save each partition standads into a file ==
    i = 0
    partitions = dict((str(k), []) for k in range(num_cls))
    for p in parts:
        name = nodes.iat[i, 0]
        i += 1
        partitions[str(p)].append(name)

    i = 0
    count = 0
    for p in partitions:
        if len(partitions[p]) == 0:
            continue
        count += len(partitions[p])
        f = open(cls_address_metis + 'clusters/cluster-' + str(i) + '.txt', 'w+')
        [f.write(l + '\n') for l in partitions[p]]
        f.close()
        i += 1


def cluster_statistics(df, cls_statistics, num_cls, cls_address):
    for c in range(num_cls):
        try:
            cured = df.loc[df.cluster == c][['response']].value_counts().cured
        except AttributeError:
            cured = 0
        try:
            non_cured = df.loc[df.cluster == c][['response']].value_counts().non_cured
        except AttributeError:
            non_cured = 0
        cls_statistics.at['cured', 'cluster-' + str(c)] = int(cured)  # / 14
        cls_statistics.at['non_cured', 'cluster-' + str(c)] = int(non_cured)  # / 73
    cls_statistics.to_csv(cls_address + 'cls_statistics.csv')


def update_cluster_folder(cls_address):
    if os.path.exists(cls_address + 'clusters/'):
        current_path = os.path.dirname(os.path.realpath(__file__))
        results_folder = glob.glob(current_path + '/' + cls_address + 'cluster*')
        for r, d, f in os.walk(results_folder[0]):
            for files in f:
                os.remove(os.path.join(r, files))
    else:
        os.makedirs(cls_address + 'clusters/')


def compute_cluster_statistic(num_cls, cls_address, sim_matrix, data_kmeans):
    cls_statistics = pd.DataFrame(columns=['cluster-' + str(x) for x in range(num_cls)], index=['cured', 'non_cured'])
    entries = os.listdir(cls_address + 'clusters/')
    for file in entries:
        sim_matrix.loc[sim_matrix.index.isin(Utility.load_cluster(file, cls_address + 'clusters/')), 'cluster'] = int(
            file[:-4].split('-')[1])
        data_kmeans.loc[data_kmeans.donor.isin(Utility.load_cluster(file, cls_address + 'clusters/')), 'cluster'] = int(
            file[:-4].split('-')[1])
    """Compute statistics for each cluster"""
    cluster_statistics(sim_matrix, cls_statistics, num_cls, cls_address)
    return sim_matrix, data_kmeans


def plot_cluster(num_cls, sim_matrix, th, data_kmeans, path_plot):
    if not os.path.exists(path_plot):
        os.makedirs(path_plot)
    if num_cls < 9:
        new_df = Utility.plot_semEP(num_cls, sim_matrix, path_plot, 'PCA_th_' + str(th) + 'matrix.pdf', scale=False)
        new_df[['response', 'cluster']].to_csv(path_plot + 'th_' + str(th) + '_summary.csv')
        Utility.plot_semEP(num_cls, data_kmeans.drop(columns=['donor']), path_plot, 'PCA_th_' + str(th) + '.pdf',
                           scale=False)


threshold = [45, 47, 50, 52, 55, 57, 60, 62, 65, 67, 70, 72, 75, 77, 80]
path_model = '../models/'
cls_measure = '/media/rivas/Data1/Projects/ImProVIT/SemEP/clusteringMeasures'


"""Load donor responses file"""
target = pd.read_csv('../1)cured_at_the_end_time_point.csv')[['donorID']]
target.replace('http://www.project-improvit.de/Donor/', 'Donor:', regex=True, inplace=True)
"""Load donors data of Kmeans"""
data_kmeans = pd.read_csv('../DescriptiveAnalysisCluster/PCA/cluster.csv')
"""Save Kmeans-Clusters"""
# for i in data_kmeans.cluster.unique():
#     data_kmeans.loc[data_kmeans.cluster==i][['donor']].to_csv('clusteringMeasures/relationalData/Kmeans/clusters/'+'cluster-'+str(i)+'.txt',
#                                                               index=None, header=None)
data_kmeans = data_kmeans.iloc[:, :-3]
file_address = 'clusteringMeasures/relationalData/'
path_plot = '../Plots/relationalData/'

for th in threshold:
    cls_address = file_address + 'SemEP_' + str(th) + '/'
    cls_address_metis = file_address + 'METIS_' + str(th) + '/'
    kmeans_address = file_address + 'Kmeans_' + str(th) + '/'

    update_cluster_folder(cls_address)
    """Create similarity matrix of Donors"""
    sim_matrix, percentile, list_sim = Utility.matrix_similarity(data_kmeans.drop(columns=['response']), Utility.cosine_sim, th)  # cosine_sim, euclidean_distance
    Utility.SemEP_structure(file_address + 'matrix_donor.tsv', sim_matrix, sep=' ')
    sim_matrix.to_csv(file_address + 'matrix_sim.txt', index=False, float_format='%.5f', mode='w+', header=False)
    Utility.create_entitie(sim_matrix.columns.to_list(), file_address + 'donor.txt')
    """Execute SemEP"""
    num_cls = call_semEP(percentile, cls_address, file_address)
    """METIS"""
    update_cluster_folder(cls_address_metis)
    if num_cls>1:
        nodes = METIS_Undirected_MAX_based_similarity_graph(sim_matrix, cls_address_metis)
        call_metis(num_cls, nodes, cls_address_metis)
    """Labeling donors in the matrix"""
    sim_matrix['response'] = 'non_cured'
    sim_matrix.loc[sim_matrix.index.isin(target.donorID), 'response'] = 'cured'
    """Compute statistics for each cluster from SemEP"""
    sim_matrix, data_kmeans = compute_cluster_statistic(num_cls, cls_address, sim_matrix, data_kmeans)
    plot_cluster(num_cls, sim_matrix, th, data_kmeans, path_plot)
    data_kmeans.drop(columns=['cluster'], inplace=True)
    sim_matrix.drop(columns=['cluster'], inplace=True)
    """Compute statistics for each cluster from METIS"""
    if num_cls > 1:
        sim_matrix, data_kmeans = compute_cluster_statistic(num_cls, cls_address_metis, sim_matrix, data_kmeans)


    data_kmeans.drop(columns=['cluster'], inplace=True)
    # sim_matrix['donor'] = sim_matrix.index
    # sim_matrix.drop(columns=['cluster'], inplace=True)
    # print(data_kmeans)
    update_cluster_folder(kmeans_address)
    # num_cls = Utility.elbow_KMeans(sim_matrix.iloc[:, :-2], 1, 15, kmeans_address)  # df_donor
    # if num_cls is None:
    #     num_cls = 15
    new_df, cls_report = Utility.plot_cluster(num_cls, data_kmeans, kmeans_address, scale=False)  # sim_matrix
    new_df.to_csv(kmeans_address + 'cluster.csv', index=None)
    # update_cluster_folder(kmeans_address)
    """Save Kmeans-Clusters"""
    for cls in range(num_cls):
        new_df.loc[new_df.cluster == cls][['donor']].to_csv(
            kmeans_address + 'clusters/' + 'cluster-' + str(cls) + '.txt', index=None, header=None)
    """Compute statistics for each cluster"""
    cls_statistics = pd.DataFrame(columns=['cluster-' + str(x) for x in range(num_cls)],
                                  index=['cured', 'non_cured'])
    cluster_statistics(new_df, cls_statistics, num_cls, kmeans_address)



"""Density of Donor Similarity"""
# Utility.density_plot(list_sim, path_plot)
    # ax = standard_similarity["similarity"].plot.kde(bw_method=0.1)
    # fig = ax.get_figure()
    # fig.savefig(path_plot + 'SimilarityDensity.pdf', format='pdf', bbox_inches='tight')
    # plt.close()

