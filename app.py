# Streamlit demo to show active learning on regression datasets. I will also show the improvement of the model on increasing pool points.

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from modAL.models import ActiveLearner, CommitteeRegressor
from modAL.disagreement import max_std_sampling
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import seaborn as sns
import base64
# set the random seed
np.random.seed(1)

st.title("Active Learning on Regression Datasets")
st.write("---")
# Button used to train the model
train_model = st.sidebar.button("Start Active Learning")

# Take inputs from the streamlit
dataset_name = st.sidebar.selectbox("Select Dataset:", ("Sinusoidal Dataset", "Absolute Dataset"))
strategy = st.sidebar.selectbox("Active Learning Strategy:", ("Gaussian Regressor Std", "Ensembled Gaussian Regressors Std"))
iterations = st.sidebar.slider("Number of Iterations:", 5, 25, 5)
pool_points = st.sidebar.slider("Number of Pool Points:", 3, 30, 5)
st.write("This demo shows the active learning on regression datasets. The model is trained on a small set of points and then the model is used to predict the labels of the remaining points. The points with the highest uncertainty are selected and added to the training set. The model is then retrained on the new training set. This process is repeated for a particular number of iterations.")
st.write("Please select the dataset, active learning strategy, number of iterations and number of pool points. Then click on the button to start the active learning.")

def GP_regression_std(regressor, X, n_instances=1):
    _, std = regressor.predict(X, return_std=True)
    # find the top n_instances points with highest uncertainty
    query_idx = np.argsort(std.ravel())[-n_instances:]
    return query_idx, X[query_idx]

def random_sampling(regressor, X, n_instances=1):
    query_idx = np.random.choice(range(len(X)), size=n_instances, replace=False)
    return query_idx, X[query_idx]

if train_model:
    if dataset_name == "Sinusoidal Dataset":
        X = np.random.choice(np.linspace(0, 6, 10000), size=200, replace=False).reshape(-1, 1)
        y = np.sin(X) + np.random.normal(scale=0.3, size=X.shape)
        X_grid = np.linspace(0, 6, 1000)
        x_ensemble = np.linspace(0, 6, 100)
    elif dataset_name == "Absolute Dataset":
        X = np.concatenate((np.random.rand(100)-1, np.random.rand(100)))
        y = np.abs(X) + np.random.normal(scale=0.2, size=X.shape)
        X_grid = np.linspace(-1, 1, 1000)
        x_ensemble = np.linspace(-1, 1, 100)
    else:
        st.write("Please select a dataset.")

    st.write("The model will be trained on the following dataset:")
    with plt.style.context('seaborn-white'):
        fig = plt.figure(figsize=(8, 4))
        plt.scatter(X, y, c='k', s=20)
        plt.title(dataset_name)
        plt.xlabel("X")
        plt.ylabel("y")
        plt.show()
        st.pyplot(fig)
    
    if strategy == "Gaussian Regressor Std":
        # initial model
        n_initial = 5
        initial_idx = np.random.choice(range(len(X)), size=n_initial, replace=False)
        X_training, y_training = X[initial_idx], y[initial_idx]
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
                + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))

        regressor = ActiveLearner(
            estimator=GaussianProcessRegressor(kernel=kernel),
            query_strategy=GP_regression_std,
            X_training=X_training.reshape(-1, 1), y_training=y_training.reshape(-1, 1)
        )
        y_pred, y_std = regressor.predict(X_grid.reshape(-1, 1), return_std=True)
        y_pred, y_std = y_pred.ravel(), y_std.ravel()
        # also use another GaussianProcessRegressor which gets trained similary and parallelly but the random points are selected 

        random_regressor = ActiveLearner(
            estimator=GaussianProcessRegressor(kernel=kernel),
            query_strategy=random_sampling,
            X_training=X_training.reshape(-1, 1), y_training=y_training.reshape(-1, 1)
        )
        st.write("Initially a few random points are selected and the model is trained on them. Below is the initial prediction of the model.")
        st.write("---")
        # show the iniial_idx points in red color
        with plt.style.context('seaborn-white'):
            fig1= plt.figure(figsize=(10, 5))
            plt.plot(X_grid, y_pred)
            plt.fill_between(X_grid, y_pred - y_std, y_pred + y_std, alpha=0.2)
            plt.scatter(X, y, c='k', s=20)
            plt.scatter(X[initial_idx], y[initial_idx], c='r', s=20, label='Initial Query points')
            plt.title('Initial prediction')
            plt.xlabel("X")
            plt.ylabel("y")
            plt.legend()
            plt.show()
            st.pyplot(fig1)

        # active learning loop
        # store the query points
        # store everything so that we can animate it later
        # also calculate the rmse and store it for animation
        y_pred_lst = []
        y_std_lst = []
        query_points = [initial_idx]
        for idx in range(iterations):
            # query pool_points points from the regressor
            query_idx, query_instance = regressor.query(X.reshape(-1, 1), n_instances=pool_points)
            # append the query_idx in the query_points list
            query_points.append(query_idx)
            # query random points from the random_regressor
            query_random_idx, query_random_instance = random_regressor.query(X.reshape(-1, 1), n_instances=pool_points)
            # teach the regressor the pool_points points it has requested
            regressor.teach(X[query_idx].reshape(-1, 1), y[query_idx].reshape(-1, 1))
            random_regressor.teach(X[query_random_idx].reshape(-1, 1), y[query_random_idx].reshape(-1, 1))
            # get the regressor predictions for the entire pool_points
            y_pred, y_std = regressor.predict(X_grid.reshape(-1, 1), return_std=True)
            y_pred, y_std = y_pred.ravel(), y_std.ravel()
            # get the random_regressor predictions for the entire pool_points
            y_pred_random, y_std_random = random_regressor.predict(X_grid.reshape(-1, 1), return_std=True)
            y_pred_random, y_std_random = y_pred_random.ravel(), y_std_random.ravel()

            y_pred_lst.append(y_pred)
            y_std_lst.append(y_std)
        
        st.write("The model is being trained iteratively on the queried points and the animation of the fit being changed is shown below.")
        # use the y_pred_lst and y_std_lst and query_points to animate the plot
        fignew, ax = plt.subplots(tight_layout=True, figsize=(8, 4))
        ax = fignew.gca()   
        def animate(i):
            ax.clear()
            ax.plot(X_grid, y_pred_lst[i], label='Prediction')
            ax.fill_between(X_grid, y_pred_lst[i] - y_std_lst[i], y_pred_lst[i] + y_std_lst[i], alpha=0.2, label='Uncertainty')
            ax.scatter(X, y, c='k', s=20, label='Data points')
            ax.scatter(X[query_points[i]], y[query_points[i]], c='r', s=20, label='Query points')
            ax.legend()
            ax.set_title(f'Iteration {i+1}')
            ax.set_xlabel("X")
            ax.set_ylabel("y")
            sns.despine()

        # Streamlit will automatically close the figure when the animation is finished
        anim = FuncAnimation(fignew, animate, frames=len(query_points)-1, interval=1000, repeat=False)
        anim.save('gaussian_process.gif', writer='PillowWriter')
        file_ = open("gaussian_process.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()

        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="bruh gif">',
            unsafe_allow_html=True,
        )
        # find the predictions of random_regressor on the X_grid
        y_pred_random, y_std_random = random_regressor.predict(X_grid.reshape(-1, 1), return_std=True)
        y_pred_random, y_std_random = y_pred_random.ravel(), y_std_random.ravel()
        st.write("---")
        st.write("We parallelly train another regressor on the same dataset but the points are selected randomly. Below is the comparision of the final fits of both the regressors.")
        # make a plot showing both fits: the one with the random points and the one with the active learning points
        # just draw the fits using plt.subplot and the predictions on the X_grid by the final regressors

        fig = plt.figure(figsize=(14, 7))
        plt.subplot(1, 2, 1)
        plt.plot(X_grid, y_pred_lst[-1], label='Prediction')
        plt.fill_between(X_grid, y_pred_lst[-1] - y_std_lst[-1], y_pred_lst[-1] + y_std_lst[-1], alpha=0.2, label='Uncertainty')
        plt.scatter(X, y, c='k', s=20, label='Data points')
        plt.legend()
        plt.title('Active Learning')
        plt.xlabel("X")
        plt.ylabel("y")

        plt.subplot(1, 2, 2)
        plt.plot(X_grid, y_pred_random, label='Prediction')
        plt.fill_between(X_grid, y_pred_random - y_std_random, y_pred_random + y_std_random, alpha=0.2, label='Uncertainty')
        plt.scatter(X, y, c='k', s=20, label='Data points')
        plt.legend()
        plt.title('Random Sampling')
        plt.xlabel("X")
        plt.ylabel("y")
        plt.show()
        st.pyplot(fig)

        st.write("The model with active learning strategy using standard deviation as the uncertainity measure performs better than the model with random sampling.")
        st.write("---")
    elif strategy == "Ensembled Gaussian Regressors Std":
        st.write("The strategy being used is dependent on the standard deviation provided by the committee / ensemble of the Gaussian regressors.")
        n_initial = 10
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
                + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))

        initial_idx = list()
        initial_idx.append(np.random.choice(range(100), size=n_initial, replace=False))
        initial_idx.append(np.random.choice(range(100, 200), size=n_initial, replace=False))
        learner_list = [ActiveLearner(
                                estimator=GaussianProcessRegressor(kernel),
                                X_training=X[idx].reshape(-1, 1), y_training=y[idx].reshape(-1, 1)
                        )
                        for idx in initial_idx]
        learner_list_random = [ActiveLearner(
                                estimator=GaussianProcessRegressor(kernel),
                                X_training=X[idx].reshape(-1, 1), y_training=y[idx].reshape(-1, 1)
                        )
                        for idx in initial_idx]
        # initializing the Committee
        committee = CommitteeRegressor(
            learner_list=learner_list,
            query_strategy=max_std_sampling
        )
        random_committee = CommitteeRegressor(
            learner_list=learner_list_random,
            query_strategy=random_sampling
        )
        # visualizing the regressors and give initial_idx as red points
        st.write("Initially a few random points are selected and the model is trained on them. Below is the initial prediction of the model.")
        st.write("As you selected ensemble of regressors, below are the predictions of each regressor and the ensemble prediction in two different plots.")
        with plt.style.context('seaborn-white'):
            fig = plt.figure(figsize=(14, 7))
            plt.subplot(1, 2, 1)
            for learner_idx, learner in enumerate(committee):
                plt.plot(x_ensemble, learner.predict(x_ensemble.reshape(-1, 1)), linewidth=5)
            plt.scatter(X, y, c='k')
            plt.scatter(X[initial_idx[0]], y[initial_idx[0]], c='r', s=20, label='Initial Query points')
            plt.title('Regressors')
            plt.legend()

            plt.subplot(1, 2, 2)
            pred, std = committee.predict(x_ensemble.reshape(-1, 1), return_std=True)
            pred = pred.reshape(-1, )
            std = std.reshape(-1, )
            plt.plot(x_ensemble, pred, c='r', linewidth=5)
            plt.fill_between(x_ensemble, pred - std, pred + std, alpha=0.2)
            plt.scatter(X, y, c='k')
            plt.scatter(X[initial_idx[0]], y[initial_idx[0]], c='r', s=20, label='Initial Query points')
            plt.title('Prediction of the ensemble')
            plt.legend()
            plt.show()
            st.pyplot(fig)

        st.write("---")
        # active learning loop
        # store the query points, predictions, std for each iterations so that we can animate it later
        query_points = []
        y_pred_lst = []
        y_std_lst = []
        for idx in range(iterations):
            # do the below steps to query pool_points points from the regressor
            query_lst = []
            for i in range(pool_points):
                query_idx, query_instance = committee.query(X.reshape(-1, 1))
                query_lst.extend(query_idx)  # Use extend to flatten the list of lists
            query_points.append(query_lst)
            # query random points from the random_regressor
            query_random_idx = np.random.choice(range(len(X)), size=pool_points, replace=False)
            random_committee.teach(X[query_random_idx].reshape(-1, 1), y[query_random_idx].reshape(-1, 1))
            # use committee.teach() to teach the regressor the pool_points points it has requested
            committee.teach(X[query_lst].reshape(-1, 1), y[query_lst].reshape(-1, 1))
            # get the regressor predictions for the entire pool_points
            y_pred, y_std = committee.predict(X_grid.reshape(-1, 1), return_std=True)
            y_pred, y_std = y_pred.ravel(), y_std.ravel()
            # get the random_regressor predictions for the entire pool_points
            y_pred_random, y_std_random = random_committee.predict(X_grid.reshape(-1, 1), return_std=True)
            y_pred_random, y_std_random = y_pred_random.ravel(), y_std_random.ravel()
            y_pred_lst.append(y_pred)
            y_std_lst.append(y_std)

        st.write("The model is being trained iteratively on the queried points and the animation of the fit being changed is shown below.")
        # use the y_pred_lst and y_std_lst and query_points to animate the plot
        fignew, ax = plt.subplots(tight_layout=True, figsize=(8, 4))
        ax = fignew.gca()   
        def animate(i):
            ax.clear()
            ax.plot(X_grid, y_pred_lst[i], label='Prediction')
            ax.fill_between(X_grid, y_pred_lst[i] - y_std_lst[i], y_pred_lst[i] + y_std_lst[i], alpha=0.2, label='Uncertainty')
            ax.scatter(X, y, c='k', s=20, label='Data points')
            ax.scatter(X[query_points[i]], y[query_points[i]], c='r', s=20, label='Query points')
            ax.legend()
            ax.set_title(f'Ensemble Prediction for Iteration {i+1}')
            ax.set_xlabel("X")
            ax.set_ylabel("y")
            sns.despine()
        
        # Streamlit will automatically close the figure when the animation is finished
        anim = FuncAnimation(fignew, animate, frames=len(query_points)-1, interval=1000, repeat=False)
        anim.save('ensembled_gaussian_process.gif', writer='PillowWriter')
        file_ = open("ensembled_gaussian_process.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()

        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="bruh gif">',
            unsafe_allow_html=True,
        )
        st.write("---")
        st.write("We parallelly train another regressor on the same dataset but the points are selected randomly. Below is the comparision of the final fits of both the regressors.")
        fig = plt.figure(figsize=(14, 7))
        plt.subplot(1, 2, 1)
        plt.plot(X_grid, y_pred_lst[-1], label='Prediction')
        plt.fill_between(X_grid, y_pred_lst[-1] - y_std_lst[-1], y_pred_lst[-1] + y_std_lst[-1], alpha=0.2, label='Uncertainty')
        plt.scatter(X, y, c='k', s=20, label='Data points')
        plt.legend()
        plt.title('Active Learning')
        plt.xlabel("X")
        plt.ylabel("y")

        plt.subplot(1, 2, 2)
        plt.plot(X_grid, y_pred_random, label='Prediction')
        plt.fill_between(X_grid, y_pred_random - y_std_random, y_pred_random + y_std_random, alpha=0.2, label='Uncertainty')
        plt.scatter(X, y, c='k', s=20, label='Data points')
        plt.legend()
        plt.title('Random Sampling')
        plt.xlabel("X")
        plt.ylabel("y")
        plt.show()
        st.pyplot(fig)
        st.write("The ensemble of regressors with uncertainity query strategy performs better than the regressor with random sampling.")
        st.write("---")
    else:
        st.write("Please select a strategy.")