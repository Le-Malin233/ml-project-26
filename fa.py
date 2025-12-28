import numpy as np
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt


# =====================================================
# 1. 生成因子分析(FA)数据
# =====================================================
def generate_fa_data(N=100, n=10, m=3, sigma2=0.1):
    """
    随机生成 x = A y + e 数据
    """
    A = np.random.randn(n, m)
    Y = np.random.randn(N, m)                  # y_t ~ N(0, I)
    noise = np.sqrt(sigma2) * np.random.randn(N, n)   # e_t ~ N(0, sigma^2 I)
    X = Y @ A.T + noise                        # 生成 x_t
    return X, A


# =====================================================
# 2. FA 模型自由参数 d_m = nm + n
# =====================================================
def num_params(n, m):
    return n*m + n     # A(n*m) + diag noise(n)


# =====================================================
# 3. 针对 m = 1..M 训练 FA 模型，计算 loglik, AIC, BIC
# =====================================================
def evaluate_models(X, M=5):
    N, n = X.shape
    results = {"m": [], "loglik": [], "AIC": [], "BIC": []}

    for m in range(1, M + 1):
        fa = FactorAnalysis(n_components=m)
        fa.fit(X)

        # sklearn 给的是平均 log-likelihood per sample
        loglik = fa.score(X) * N

        dm = num_params(n, m)
        AIC = loglik - dm
        BIC = loglik - dm * np.log(N) / 2

        results["m"].append(m)
        results["loglik"].append(loglik)
        results["AIC"].append(AIC)
        results["BIC"].append(BIC)

    return results


# =====================================================
# 4. 选择最佳 m
# =====================================================
def select_best_m(results):
    m_AIC = results["m"][np.argmax(results["AIC"])]
    m_BIC = results["m"][np.argmax(results["BIC"])]
    return m_AIC, m_BIC


# =====================================================
# 5. 单次实验可视化
# =====================================================
def plot_results(results, true_m):
    ms = results["m"]

    plt.figure(figsize=(12, 4))

    # ---- log-likelihood ----
    plt.subplot(1, 3, 1)
    plt.plot(ms, results["loglik"], marker='o')
    plt.title("Log-likelihood vs m")
    plt.xlabel("m")
    plt.ylabel("loglik")

    # ---- AIC ----
    plt.subplot(1, 3, 2)
    plt.plot(ms, results["AIC"], marker='o')
    plt.axvline(true_m, color='r', linestyle='--')
    plt.title("AIC vs m")
    plt.xlabel("m")

    # ---- BIC ----
    plt.subplot(1, 3, 3)
    plt.plot(ms, results["BIC"], marker='o')
    plt.axvline(true_m, color='r', linestyle='--')
    plt.title("BIC vs m")
    plt.xlabel("m")

    plt.tight_layout()
    plt.show()


# =====================================================
# 6. 多次实验统计 AIC/BIC 的正确率
# =====================================================
def multiple_experiments(
        trials=30, N=100, n=10, true_m=3, sigma2=0.1, M=5):

    AIC_correct = 0
    BIC_correct = 0
    AIC_results = []  # store individual trial results
    BIC_results = []

    for i in range(trials):
        X, A = generate_fa_data(N, n, true_m, sigma2)
        results = evaluate_models(X, M)
        m_AIC, m_BIC = select_best_m(results)

        is_AIC_ok = (m_AIC == true_m)
        is_BIC_ok = (m_BIC == true_m)

        AIC_results.append(is_AIC_ok)
        BIC_results.append(is_BIC_ok)

        if is_AIC_ok:
            AIC_correct += 1
        if is_BIC_ok:
            BIC_correct += 1

    AIC_wrong = trials - AIC_correct
    BIC_wrong = trials - BIC_correct

    return {
        "trials": trials,
        "AIC_correct": AIC_correct,
        "AIC_wrong": AIC_wrong,
        "BIC_correct": BIC_correct,
        "BIC_wrong": BIC_wrong,
        "AIC_accuracy": AIC_correct / trials,
        "BIC_accuracy": BIC_correct / trials,
        "AIC_results": AIC_results,
        "BIC_results": BIC_results
    }


# =====================================================
# 7. 主程序
# =====================================================
if __name__ == "__main__":

    # ---- 单次试验 ----
    print("Running one experiment ...")
    true_m = 5
    X, A = generate_fa_data(N=20, n=15, m=true_m, sigma2=0.1)
    results = evaluate_models(X, M=9)
    m_AIC, m_BIC = select_best_m(results)

    print("AIC 选择的 m =", m_AIC)
    print("BIC 选择的 m =", m_BIC)

    plot_results(results, true_m)

    # ---- 多次实验 ----
    stats = multiple_experiments(
        trials=200, N=20, n=15, true_m=true_m, sigma2=0.1, M=9
    )

    print("Total trials:", stats["trials"])
    print("AIC: 正确 / 错误 = {}/{}".format(
        stats["AIC_correct"], stats["AIC_wrong"]))
    print("BIC: 正确 / 错误 = {}/{}".format(
        stats["BIC_correct"], stats["BIC_wrong"]))
    print("AIC 准确率: {:.1f}%".format(stats["AIC_accuracy"]*100))
    print("BIC 准确率: {:.1f}%".format(stats["BIC_accuracy"]*100))
