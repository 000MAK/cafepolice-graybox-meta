# Rakamların Diliyle Güven: Graybox GLM + Metaheuristic ile Saf Prim
*Yorumlanabilir frekans–şiddet modeli ve genetik algoritmayla regularization arayışı*

*Mehmet Ali Kaya · Veri Yönetimi Uzmanı, Yönetim Bilişim Sistemleri Doktora Adayı, İstatistik Yüksek Lisans*

**Bir kasko benzeri portföyün saf primini kestirmek için frekansı Poisson, şiddeti Gamma ile modelledim; iki modelin regularization katsayısını grid taraması yerine genetik algoritmayla aradım ve modeli zamandan kopuk bir test diliminde ölçtüm. Sonuç şu: yorumlanabilir bir model, doğru bölünmüş bir veri setiyle, opak bir aramadan daha güvenilir konuşabiliyor.**

## Veri: ne kullandım, neyi uydurmadım

Bu çalışmanın dürüstlük ilkesi ilk satırdan başlıyor. freMTPL2 (Fransız motorlu taşıt üçüncü şahıs hasar verisi, açık lisanslı CC0) indirilebilirse frekans tarafında kullanılmak üzere koda gömülü; ancak freMTPL2'nin şiddet bileşeni ayrı bir dosyada ve poliçe bazında birleştirilemez olduğundan, uçtan uca gösterim **açıkça etiketlenmiş sentetik bir demo verisiyle** yapıldı. Veri üretim süreci biliniyor: her poliçe için sürücü yaşı, araç gücü, araç yaşı, bonus–malus seviyesi, trafik yoğunluğu, marka, yakıt, bölge ve bölge kodu üretildi; hasar sayısı bu özelliklere bağlı bir Poisson oranıyla, hasar tutarı ise aynı özelliklere bağlı bir Gamma ortalamasıyla türetildi (tohum 42, 20.000 poliçe). Sentetik olduğu her çıktıda yazılı; hiçbir yerde gerçek veriymiş gibi sunulmadı.

## Model: iki beyaz kutu + bir kara kutu arayış

Sigortacılığın klasik "graybox" tarifi burada iki katmanlı. Birinci katman **tamamen yorumlanabilir**: frekans için log-link'li Poisson regresyon (hasar sayısı), şiddet için log-link'li Gamma regresyon (hasar tutarı). Saf prim, bu ikisinin çarpımı — yani `beklenen hasar sıklığı × beklenen hasar şiddeti`. Aktüeryal standart formül; her katsayı tek tek okunabilir, denetlenebilir, itiraz edilebilir.

İkinci katman **opak ama kontrollü**: her iki GLM'in de bir `alpha` (L2 regularization) hiperparametresi var. Bunu sabit değerler üzerinde grid taramasıyla değil, bir **genetik algoritmayla** aradım. Kromozom iki genden oluşuyor — `log10(alpha_frekans)` ve `log10(alpha_şiddet)`. Her bireyin uygunluğu, validation dilimindeki toplam deviance (Poisson + Gamma) ile ölçülüyor; turnuva seçimi, harmanlama çaprazlaması ve Gauss mutasyonuyla 12 nesil boyunca evriliyor. Grid'in aksine, arama uzayını sürekli ve uyarlanabilir biçimde tarıyor.

## Doğrulama: zamanın dışına çıkmak

Modeli üçe böldüm: %70 eğitim, %15 doğrulama (hiperparametre seçimi), %15 test (son, tek seferlik ölçüm). Test dilimi ne eğitime ne de hiperparametre arayışına dokundu; bu yüzden sonuç, modelin hiç görmediği poliçelere karşı bir "zaman dışı" (Out-of-Time) sınamasıdır. Bölme tohumu sabit, dolayısıyla her çalıştırmada aynı poliçeler aynı dilime düşer; sonuç tekrarlanabilir.

## Bulgular

Genetik algoritma `alpha_frekans ≈ 8,06` ve `alpha_şiddet ≈ 0,23` noktasına yakınsadı. Frekans tarafının güçlü bir düzenlileştirme istemesi anlamlı: hasar sıklığı sinyali zayıf (test diliminde %4,5) ve 24 özellik içinde yalnızca birkaçı gerçekten işe yarıyor; güçlü shrinkage, gürültüyü ezerek saf primi iyileştiriyor. Şiddet tarafı ise daha az düzenlileştirme istiyor — hasar tutarının sinyali görece güçlü.

| Ölçü (test, hiç görülmemiş 3.000 poliçe) | Değer |
|---|---|
| Hasar sıklığı (gerçekleşen) | %4,51 |
| Frekans — Poisson deviance | 0,2297 |
| Şiddet — MAE | 1.235 birim |
| Şiddet — MSE | 2.652.890 |
| Saf prim — MAE | 168,7 birim |
| Saf prim — MSE | 403.411 |
| Saf prim ortalaması (tahmin) | 76,9 birim |

## Tek ders

Graybox'ın gücü, "kara kutu" kısmını bile denetlenebilir tutmasında. Alpha'yı genetik algoritmayla aramak, grid taramasına kıyasla daha az hesaplamayla daha geniş bir uzayı tarar; ama nihai karar her zaman yorumlanabilir iki GLM'in üzerine oturur. Frekans tarafı güçlü, şiddet tarafı hafif düzenlileştirme istiyor — bu da modelin iki ayrı doğasını tek bir "sihirli" parametreye sığdıramayacağımızı gösteriyor. Veri sentetikse sentetik denir; rakamlar ancak o zaman güven kurar.

- Kod: https://github.com/000MAK/cafepolice-graybox-meta
- Model kartı: https://huggingface.co/makactuary/cafepolice-graybox-meta
