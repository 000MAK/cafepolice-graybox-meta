#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graybox_engine.py — "Café Poliçe" graybox (white-box GLM) + metaheuristik (GA) saf prim motoru.

NEDİR
  * frekans  -> sklearn PoissonRegressor (log-link)   [aktüeryal standart]
  * şiddet   -> sklearn GammaRegressor   (log-link)   [aktüeryal standart]
  * saf prim  = beklenen frekans oranı x beklenen şiddet
  * metaheuristik -> GA ile iki GLM'in alpha (L2 regularization) hiperparametresi
    validation üzerinde deviance minimize edilerek aranır (GRID yok).

VERİ (dürüst etiketleme)
  * freMTPL2 (açık, CC0): frekans = ClaimNb/Exposure (freMTPL2freq), şiddet =
    ClaimAmount (freMTPL2sev). Şiddet verisi yalnız IDpol+ClaimAmount taşır; özellikler
    frekans verisinden IDpol ile birleştirilir.
  * İndirilemezse SENTETİK demo (sabit seed, açıkça etiketli). Kaynak her çıktıda yazılır.

ÇIKTI (repo içinde, C:'ye YAZILMAZ)
  * outputs/metrics.json, outputs/sonuc_tablosu.csv
  * outputs/charts/01_frekans_fit.png, 02_siddet_fit.png, 03_ga_yakinsama.png

Kullanım:
  python src/graybox_engine.py            -> gerçek veriyi dene (freMTPL2), olmazsa sentetik
  python src/graybox_engine.py --test     -> ağsız, deterministik kendini doğrula
  python src/graybox_engine.py --real     -> freMTPL2'yi zorla dene
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_gamma_deviance,
    mean_poisson_deviance,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
KOK = Path(__file__).resolve().parent.parent
CIKTI = KOK / "outputs"
GRAFIK = CIKTI / "charts"
VERI_DIZIN = KOK / "data"

# freMTPL2 ile ortak, yorumlanabilir özellik kümesi
SAYISAL = ["DrivAge", "VehPower", "VehAge", "BonusMalus", "Density"]
KATEGORIK = ["VehBrand", "VehGas", "Area", "Region"]
OZELLIKLER = SAYISAL + KATEGORIK


# ------------------------------------------------------------------ veri
def synthetic_data(n: int = 20000, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """SENTETİK demo verisi (açıkça etiketli). Gerçek veri DEĞİL."""
    rng = np.random.default_rng(seed)
    driv_age = rng.integers(18, 90, n).astype(float)
    veh_power = rng.integers(4, 15, n).astype(float)
    veh_age = rng.integers(0, 20, n).astype(float)
    bonus_malus = rng.integers(50, 150, n).astype(float)
    density = rng.integers(1, 11, n).astype(float)
    veh_brand = rng.integers(1, 7, n).astype(str)
    veh_gas = rng.choice(["Regular", "Diesel"], n)
    area = rng.choice(["A", "B", "C", "D", "E", "F"], n)
    region = rng.choice([f"R{i}" for i in (11, 21, 22, 23, 24, 25, 26, 31, 41)], n)
    exposure = rng.uniform(0.5, 1.0, n)

    log_rate = (-3.0 + 0.35 * (driv_age < 25).astype(float)
                - 0.010 * (driv_age - 40) / 10.0
                + 0.012 * (bonus_malus - 100) / 10.0
                + 0.020 * (density / 10.0))
    rate = np.exp(log_rate)
    claim_nb = rng.poisson(rate * exposure)

    log_sev = (7.5 + 0.10 * (veh_power / 10.0)
               + 0.03 * (veh_age / 10.0)
               + 0.04 * (bonus_malus - 100) / 10.0)
    mean_sev = np.exp(log_sev)
    phi = 0.5
    avg_claim = np.where(claim_nb > 0,
                         rng.gamma(shape=1.0 / phi, scale=mean_sev * phi), 0.0)

    df = pd.DataFrame({
        "DrivAge": driv_age, "VehPower": veh_power, "VehAge": veh_age,
        "BonusMalus": bonus_malus, "Density": density, "VehBrand": veh_brand,
        "VehGas": veh_gas, "Area": area, "Region": region,
        "Exposure": exposure, "ClaimNb": claim_nb, "AvgClaimAmount": avg_claim,
    })
    freq_df = df[OZELLIKLER + ["Exposure", "ClaimNb"]].copy()
    sev_df = df[df["ClaimNb"] > 0][OZELLIKLER + ["AvgClaimAmount"]].rename(
        columns={"AvgClaimAmount": "ClaimAmount"})
    return freq_df, sev_df, f"sentetik-demo (seed={seed})"


def load_fremtpl2(subsample: int = 50000) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """freMTPL2 (açık, CC0) — frekans (ClaimNb/Exposure) + şiddet (ClaimAmount).

    Şiddet verisi yalnız IDpol+ClaimAmount taşır; özellikler frekans verisinden
    IDpol ile birleştirilir. Frekans tarafı bellek/süre için alt-örneklenir (etiketli).
    """
    from sklearn.datasets import fetch_openml
    VERI_DIZIN.mkdir(parents=True, exist_ok=True)
    freq = fetch_openml("freMTPL2freq", version=1, as_frame=True, parser="auto",
                        data_home=str(VERI_DIZIN)).frame
    sev = fetch_openml("freMTPL2sev", version=1, as_frame=True, parser="auto",
                       data_home=str(VERI_DIZIN)).frame
    # şiddet: TÜM frekans özellikleriyle birleştir (alt-örnek DEĞİL, aksi halde NaN)
    sev = sev.merge(freq[["IDpol"] + OZELLIKLER], on="IDpol", how="left")
    sev = sev.dropna(subset=OZELLIKLER + ["ClaimAmount"])
    # frekans: alt-örnek (dürüst etiket)
    freq = freq.sample(n=min(subsample, len(freq)), random_state=SEED)
    freq_df = freq[OZELLIKLER + ["Exposure", "ClaimNb"]].copy()
    sev_df = sev[OZELLIKLER + ["ClaimAmount"]].copy()
    sev_df = sev_df[sev_df["ClaimAmount"] > 0]
    etiket = (f"freMTPL2 (CC0, openml) — frekans alt-örnek n={len(freq_df)}, "
              f"şiddet n={len(sev_df)}")
    return freq_df, sev_df, etiket


def load_data(use_real: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Graybox verisi + dürüst kaynak etiketi. Önce gerçek (freMTPL2), olmazsa sentetik."""
    if use_real:
        try:
            return load_fremtpl2()
        except Exception:
            return synthetic_data(seed=SEED)
    return synthetic_data(seed=SEED)


# ------------------------------------------------------------------ özellik/GLM
def _encode(freq_df: pd.DataFrame, sev_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Ortak one-hot kodlama + sayısal standartlaştırma (frekans + şiddet aynı uzayda)."""
    birlesik = pd.concat([freq_df[OZELLIKLER], sev_df[OZELLIKLER]], axis=0, ignore_index=True)
    say = birlesik[SAYISAL].to_numpy(dtype=float)
    ort, ss = say.mean(axis=0), say.std(axis=0)
    ss[ss == 0] = 1.0
    say = (say - ort) / ss
    kat = pd.get_dummies(birlesik[KATEGORIK], drop_first=True).to_numpy(dtype=float)
    kodlu = np.hstack([say, kat])
    n_f = len(freq_df)
    return kodlu[:n_f], kodlu[n_f:]


def _split(n: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """70 / 15 / 15 bölme (random_state sabit)."""
    idx = np.arange(n)
    itr, ite = train_test_split(idx, test_size=0.15, random_state=seed)
    itr, iva = train_test_split(itr, test_size=0.15 / 0.85, random_state=seed)
    return itr, iva, ite


def _fit_freq(X, y_freq, exposure, alpha) -> PoissonRegressor:
    m = PoissonRegressor(alpha=alpha, max_iter=1000)
    m.fit(X, y_freq, sample_weight=exposure)
    return m


def _fit_sev(X_sev, y_sev, alpha) -> GammaRegressor:
    m = GammaRegressor(alpha=alpha, max_iter=1000)
    m.fit(X_sev, y_sev)
    return m


def ga_optimize(Xtr, yf_tr, w_tr, Xs_tr, ys_tr,
                Xva, yf_va, w_va, Xs_va, ys_va,
                nesil: int = 10, pop: int = 16, seed: int = SEED) -> dict:
    """GA: iki GLM'in alpha'sını validation deviance'ı üzerinde arar (GRID yok)."""
    rng = random.Random(seed)
    lo, hi = -6.0, 3.0
    _cache: dict = {}
    izleme: list = []

    def hedef(krom):
        key = (round(krom[0], 6), round(krom[1], 6))
        if key in _cache:
            return _cache[key]
        af, ase = 10.0 ** krom[0], 10.0 ** krom[1]
        mf = _fit_freq(Xtr, yf_tr, w_tr, af)
        pf_va = mf.predict(Xva) * w_va
        fdev = float(mean_poisson_deviance(yf_va, pf_va))
        ms = _fit_sev(Xs_tr, ys_tr, ase)
        ps_va = ms.predict(Xs_va)
        gdev = float(mean_gamma_deviance(ys_va, ps_va))
        deger = fdev + gdev
        _cache[key] = deger
        return deger

    def rastgele():
        return [rng.uniform(lo, hi), rng.uniform(lo, hi)]

    pop_ = [rastgele() for _ in range(pop)]

    for gen in range(nesil):
        elit = sorted(pop_, key=hedef)[:2]
        izleme.append((gen, hedef(elit[0])))
        yeni = [e[:] for e in elit]
        while len(yeni) < pop:
            a = min(rng.sample(pop_, 3), key=hedef)
            b = min(rng.sample(pop_, 3), key=hedef)
            cocuk = [rng.uniform(min(a[i], b[i]), max(a[i], b[i])) for i in range(2)]
            cocuk = [min(max(c + rng.gauss(0, 0.15), lo), hi) for c in cocuk]
            yeni.append(cocuk)
        pop_ = yeni

    en_iyi = min(pop_, key=hedef)
    izleme.append((nesil, hedef(en_iyi)))
    return {
        "alpha_frekans": float(10.0 ** en_iyi[0]),
        "alpha_siddet": float(10.0 ** en_iyi[1]),
        "nesil": nesil, "populasyon": pop,
        "log10_alpha_frekans": float(en_iyi[0]),
        "log10_alpha_siddet": float(en_iyi[1]),
        "yakinsama": [[int(g), round(float(f), 6)] for g, f in izleme],
    }


# ------------------------------------------------------------------ grafik
def _bin_uyum(gercek, tahmin, adet=10):
    siralama = np.argsort(tahmin)
    gercek = np.asarray(gercek, dtype=float)[siralama]
    tahmin = np.asarray(tahmin, dtype=float)[siralama]
    n = len(tahmin)
    sinirlar = [int(i * n / adet) for i in range(adet + 1)]
    g, t = [], []
    for i in range(adet):
        a, b = sinirlar[i], sinirlar[i + 1]
        if a < b:
            g.append(float(np.mean(gercek[a:b])))
            t.append(float(np.mean(tahmin[a:b])))
    return t, g


def cizimler(rate_te, yf_te, w_te, sev_pred, ys, yakinsama) -> dict:
    GRAFIK.mkdir(parents=True, exist_ok=True)
    yollar = {}

    gozlenen_oran = yf_te / w_te
    tx, ty = _bin_uyum(gozlenen_oran, rate_te)
    plt.figure(figsize=(5, 4))
    plt.scatter(tx, ty, color="#1F4E79", zorder=3)
    lim = [min(tx + ty), max(tx + ty)]
    plt.plot(lim, lim, "--", color="#888")
    plt.xlabel("Tahmin edilen frekans oranı (yıl başına)")
    plt.ylabel("Gözlenen frekans oranı")
    plt.title("Frekans GLM uyumu (test)")
    plt.tight_layout()
    yollar["01_frekans_fit"] = GRAFIK / "01_frekans_fit.png"
    plt.savefig(yollar["01_frekans_fit"], dpi=110)
    plt.close()

    sx, sy = _bin_uyum(ys, sev_pred)
    plt.figure(figsize=(5, 4))
    plt.scatter(sx, sy, color="#C00000", zorder=3)
    lim = [min(sx + sy), max(sx + sy)]
    plt.plot(lim, lim, "--", color="#888")
    plt.xlabel("Tahmin edilen şiddet")
    plt.ylabel("Gözlenen şiddet")
    plt.title("Gamma şiddet uyumu (test)")
    plt.tight_layout()
    yollar["02_siddet_fit"] = GRAFIK / "02_siddet_fit.png"
    plt.savefig(yollar["02_siddet_fit"], dpi=110)
    plt.close()

    gen = [p[0] for p in yakinsama]
    fit = [p[1] for p in yakinsama]
    plt.figure(figsize=(5, 4))
    plt.plot(gen, fit, "-o", color="#2E75B6")
    plt.xlabel("Nesil")
    plt.ylabel("En iyi uygunluk (toplam deviance)")
    plt.title("GA yakınsaması")
    plt.tight_layout()
    yollar["03_ga_yakinsama"] = GRAFIK / "03_ga_yakinsama.png"
    plt.savefig(yollar["03_ga_yakinsama"], dpi=110)
    plt.close()

    return {k: str(v) for k, v in yollar.items()}


# ------------------------------------------------------------------ çıktı
def yaz_cikti(sonuclar: dict) -> None:
    CIKTI.mkdir(parents=True, exist_ok=True)
    (CIKTI / "metrics.json").write_text(
        json.dumps(sonuclar, ensure_ascii=False, indent=2), encoding="utf-8")
    m = sonuclar["metrikler"]
    satirlar = [
        ("veri", sonuclar["veri"]),
        ("frekans_satir", sonuclar["n_frekans"]),
        ("siddet_satir", sonuclar["n_siddet"]),
        ("split_train", sonuclar["split"]["train_oran"]),
        ("split_val", sonuclar["split"]["val_oran"]),
        ("split_test", sonuclar["split"]["test_oran"]),
        ("alpha_frekans", sonuclar["ga"]["alpha_frekans"]),
        ("alpha_siddet", sonuclar["ga"]["alpha_siddet"]),
        ("frekans_poisson_deviance_test", m["frekans_poisson_deviance_test"]),
        ("siddet_mae_test", m["siddet_mae_test"]),
        ("siddet_mse_test", m["siddet_mse_test"]),
        ("saf_prim_ort_test", m["saf_prim_ort_test"]),
        ("hasar_sikligi_test", m["hasar_sikligi_test"]),
    ]
    pd.DataFrame(satirlar, columns=["alan", "deger"]).to_csv(
        CIKTI / "sonuc_tablosu.csv", index=False, encoding="utf-8")


# ------------------------------------------------------------------ ana akış
def run(use_real: bool = True, seed: int = SEED) -> dict:
    freq_df, sev_df, veri_etiket = load_data(use_real=use_real)
    X_freq, X_sev = _encode(freq_df, sev_df)
    y_freq = freq_df["ClaimNb"].to_numpy(dtype=float)
    exposure = freq_df["Exposure"].to_numpy(dtype=float)
    y_sev = sev_df["ClaimAmount"].to_numpy(dtype=float)
    n_f, n_s = len(freq_df), len(sev_df)

    itr_f, iva_f, ite_f = _split(n_f, seed)
    itr_s, iva_s, ite_s = _split(n_s, seed)

    Xtr = X_freq[itr_f]; yf_tr = y_freq[itr_f]; w_tr = exposure[itr_f]
    Xva = X_freq[iva_f]; yf_va = y_freq[iva_f]; w_va = exposure[iva_f]
    Xte = X_freq[ite_f]; yf_te = y_freq[ite_f]; w_te = exposure[ite_f]
    Xs_tr = X_sev[itr_s]; ys_tr = y_sev[itr_s]
    Xs_va = X_sev[iva_s]; ys_va = y_sev[iva_s]
    Xs_te = X_sev[ite_s]; ys_te = y_sev[ite_s]

    ga = ga_optimize(Xtr, yf_tr, w_tr, Xs_tr, ys_tr,
                     Xva, yf_va, w_va, Xs_va, ys_va, seed=seed)

    mf = _fit_freq(Xtr, yf_tr, w_tr, ga["alpha_frekans"])
    ms = _fit_sev(Xs_tr, ys_tr, ga["alpha_siddet"])

    rate_te = mf.predict(Xte)                       # beklenen frekans oranı
    freq_pred_te = rate_te * w_te                   # beklenen hasar sayısı
    fdev = float(mean_poisson_deviance(yf_te, freq_pred_te))
    sev_pred_te = ms.predict(Xs_te)
    sev_mse = float(mean_squared_error(ys_te, sev_pred_te))
    sev_mae = float(mean_absolute_error(ys_te, sev_pred_te))

    saf_prim_ort = float(np.mean(rate_te) * np.mean(sev_pred_te))
    metrikler = {
        "frekans_poisson_deviance_test": round(fdev, 6),
        "siddet_mse_test": round(sev_mse, 4),
        "siddet_mae_test": round(sev_mae, 4),
        "saf_prim_ort_test": round(saf_prim_ort, 4),
        "hasar_sikligi_test": round(float(np.mean(yf_te / w_te)), 4),
    }

    yollar = cizimler(rate_te, yf_te, w_te, sev_pred_te, ys_te, ga["yakinsama"])

    sonuclar = {
        "veri": veri_etiket,
        "n_frekans": n_f, "n_siddet": n_s,
        "split": {"train_oran": 0.7, "val_oran": 0.15, "test_oran": 0.15,
                  "n_train_freq": int(len(Xtr)), "n_val_freq": int(len(Xva)),
                  "n_test_freq": int(len(Xte)), "n_test_siddet": int(len(Xs_te))},
        "ga": ga,
        "metrikler": metrikler,
        "grafikler": yollar,
    }
    yaz_cikti(sonuclar)
    return sonuclar


# ------------------------------------------------------------------ test
def test() -> int:
    kontrols = []

    def kontrol(ad, kosul, detay=""):
        kontrols.append((ad, bool(kosul), detay))

    r = run(use_real=False, seed=SEED)
    n = r["n_frekans"]
    s = r["split"]

    kontrol("split: train ~0.70", abs(s["n_train_freq"] / n - 0.70) < 0.02)
    kontrol("split: val ~0.15", abs(s["n_val_freq"] / n - 0.15) < 0.02)
    kontrol("split: test ~0.15", abs(s["n_test_freq"] / n - 0.15) < 0.02)

    ga = r["ga"]
    kontrol("GA: alpha_frekans>0", ga["alpha_frekans"] > 0)
    kontrol("GA: alpha_siddet>0", ga["alpha_siddet"] > 0)
    kontrol("GA: yakınsama kaydedildi", len(ga["yakinsama"]) >= ga["nesil"])

    m = r["metrikler"]
    kontrol("metrik: poisson deviance", m["frekans_poisson_deviance_test"] >= 0)
    kontrol("metrik: MSE/MAE hesaplandı",
            "siddet_mse_test" in m and "siddet_mae_test" in m)
    kontrol("metrik: saf prim ort hesaplandı", m["saf_prim_ort_test"] > 0)

    kontrol("veri: sentetik açıkça etiketli", "sentetik" in r["veri"].lower())

    for k in r["grafikler"].values():
        kontrol(f"grafik var: {Path(k).name}", Path(k).exists())
    kontrol("metrics.json var", (CIKTI / "metrics.json").exists())
    kontrol("sonuc_tablosu.csv var", (CIKTI / "sonuc_tablosu.csv").exists())

    ok = sum(1 for _, k, _ in kontrols if k)
    kal = len(kontrols) - ok
    for ad, k, detay in kontrols:
        print(f"  {'[OK]    ' if k else '[KALDI] '}{ad}" + (f"  {detay}" if detay else ""))
    print(f"SONUC: {ok} başarılı, {kal} KALDI · veri={r['veri']}")
    return 0 if kal == 0 else 1


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--test" in args:
        return test()
    r = run(use_real=True)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
