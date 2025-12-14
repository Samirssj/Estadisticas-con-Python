#!/usr/bin/env python3
"""
Aplicación GUI en Tkinter para análisis estadístico y gráficos a partir de datos tabulares.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from io import StringIO
from scipy import stats
import statsmodels.api as sm
import os
from scipy.stats import binom, poisson, norm

plt.rcParams["figure.figsize"] = (6,4)

class EstadisticaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Estadística - GUI (Tkinter) - Versión Mejorada")
        self.df = pd.DataFrame()
        self.fig = None
        self.create_widgets()

    def create_widgets(self):
        frm_top = ttk.Frame(self.root)
        frm_top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        btn_load = ttk.Button(frm_top, text="Cargar CSV/XLSX", command=self.load_file)
        btn_load.pack(side=tk.LEFT, padx=4)
        btn_paste = ttk.Button(frm_top, text="Pegar tabla (desde Excel)", command=self.paste_table)
        btn_paste.pack(side=tk.LEFT, padx=4)
        btn_save = ttk.Button(frm_top, text="Exportar a Excel", command=self.export_excel)
        btn_save.pack(side=tk.LEFT, padx=4)
        btn_export_txt = ttk.Button(frm_top, text="Exportar análisis (TXT)", command=self.export_analysis_txt)
        btn_export_txt.pack(side=tk.LEFT, padx=4)
        btn_clear = ttk.Button(frm_top, text="Limpiar datos", command=self.clear_data)
        btn_clear.pack(side=tk.LEFT, padx=4)
        btn_save_dataset = ttk.Button(frm_top, text="Guardar dataset actual (xlsx)", command=self.save_dataset_current)
        btn_save_dataset.pack(side=tk.LEFT, padx=4)

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill=tk.BOTH, expand=True)

        tab_data = ttk.Frame(self.nb)
        self.nb.add(tab_data, text="Datos")
        self.table = ttk.Treeview(tab_data)
        self.table.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        sc = ttk.Scrollbar(tab_data, orient="vertical", command=self.table.yview)
        sc.pack(side=tk.RIGHT, fill=tk.Y)
        self.table.configure(yscrollcommand=sc.set)

        tab_stats = ttk.Frame(self.nb)
        self.nb.add(tab_stats, text="Estadísticos")
        frm_stats = ttk.Frame(tab_stats)
        frm_stats.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        btn_desc = ttk.Button(frm_stats, text="Descriptivos (numéricas)", command=self.show_descriptive)
        btn_desc.pack(anchor=tk.W, pady=2)
        btn_freq = ttk.Button(frm_stats, text="Tablas de frecuencia (categóricas)", command=self.show_frequencies)
        btn_freq.pack(anchor=tk.W, pady=2)
        btn_corr = ttk.Button(frm_stats, text="Correlación (Pearson/Spearman)", command=self.show_correlation)
        btn_corr.pack(anchor=tk.W, pady=2)
        btn_corrmat = ttk.Button(frm_stats, text="Matrix de correlación (gráfica)", command=self.plot_corr_matrix)
        btn_corrmat.pack(anchor=tk.W, pady=2)
        btn_reg = ttk.Button(frm_stats, text="Regresión lineal (y ~ X1 + X2 ...)", command=self.regression_dialog)
        btn_reg.pack(anchor=tk.W, pady=2)
        btn_tests = ttk.Button(frm_stats, text="Pruebas (t, chi2, ANOVA)", command=self.tests_dialog)
        btn_tests.pack(anchor=tk.W, pady=2)

        tab_plot = ttk.Frame(self.nb)
        self.nb.add(tab_plot, text="Gráficos")
        frm_plot = ttk.Frame(tab_plot)
        frm_plot.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        ttk.Label(frm_plot, text="Selecciona variable(s) y tipo de gráfico:").pack(anchor=tk.W)
        self.pl_var1 = ttk.Combobox(frm_plot, values=[])
        self.pl_var1.pack(anchor=tk.W, pady=2)
        self.pl_var2 = ttk.Combobox(frm_plot, values=[])
        self.pl_var2.pack(anchor=tk.W, pady=2)
        btn_hist = ttk.Button(frm_plot, text="Histograma", command=self.plot_hist)
        btn_hist.pack(anchor=tk.W, pady=2)
        btn_box = ttk.Button(frm_plot, text="Boxplot", command=self.plot_box)
        btn_box.pack(anchor=tk.W, pady=2)
        btn_scatter = ttk.Button(frm_plot, text="Scatter (X vs Y)", command=self.plot_scatter)
        btn_scatter.pack(anchor=tk.W, pady=2)
        btn_bar = ttk.Button(frm_plot, text="Gráfico de Barras (categoría)", command=self.plot_bar)
        btn_bar.pack(anchor=tk.W, pady=2)
        btn_pie = ttk.Button(frm_plot, text="Gráfico Circular (pie)", command=self.plot_pie)
        btn_pie.pack(anchor=tk.W, pady=2)
        btn_binom = ttk.Button(frm_plot, text="Distribución Binomial (modelo)", command=self.plot_binomial_dialog)
        btn_binom.pack(anchor=tk.W, pady=2)
        btn_poisson = ttk.Button(frm_plot, text="Distribución Poisson (modelo)", command=self.plot_poisson_dialog)
        btn_poisson.pack(anchor=tk.W, pady=2)
        btn_norm = ttk.Button(frm_plot, text="Distribución Normal (sombrear rango)", command=self.plot_normal_dialog)
        btn_norm.pack(anchor=tk.W, pady=2)
        btn_corrplt = ttk.Button(frm_plot, text="Guardar gráfico actual", command=self.save_current_fig)
        btn_corrplt.pack(anchor=tk.W, pady=6)

    # ---------- IO ----------
    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv"),("Excel files","*.xlsx;*.xls"),("All files","*.*")])
        if not path:
            return
        try:
            if path.lower().endswith(('.xls','.xlsx')):
                self.df = pd.read_excel(path)
            else:
                self.df = pd.read_csv(path)
            self.refresh_table()
            messagebox.showinfo("Cargado", f"Datos cargados: {os.path.basename(path)} (filas: {len(self.df)})")
        except Exception as e:
            messagebox.showerror("Error al cargar", str(e))

    def paste_table(self):
        try:
            s = self.root.clipboard_get()
        except Exception:
            messagebox.showerror("Portapapeles vacío", "Copia primero la tabla (CTRL+C) desde Excel/Sheets y vuelve a intentarlo.")
            return
        try:
            df = pd.read_csv(StringIO(s), sep=None, engine='python')
        except Exception:
            try:
                df = pd.read_table(StringIO(s), sep=None, engine='python')
            except Exception as e:
                messagebox.showerror("Error", "No pude interpretar el portapapeles como tabla. Asegúrate de copiar desde Excel/Sheets.")
                return
        self.df = df
        self.refresh_table()
        messagebox.showinfo("Pegado", f"Tabla pegada (filas: {len(self.df)})")

    def export_excel(self):
        if self.df.empty:
            messagebox.showwarning("Sin datos", "No hay datos para exportar.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if not path:
            return
        try:
            self.df.to_excel(path, index=False)
            messagebox.showinfo("Exportado", f"Datos exportados a {path}")
        except Exception as e:
            messagebox.showerror("Error al exportar", str(e))

    def save_dataset_current(self):
        if self.df.empty:
            messagebox.showwarning("Sin datos", "No hay datos para guardar.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if not path:
            return
        try:
            self.df.to_excel(path, index=False)
            messagebox.showinfo("Guardado", f"Dataset guardado en {path}")
        except Exception as e:
            messagebox.showerror("Error al guardar", str(e))

    def export_analysis_txt(self):
        if self.df.empty:
            messagebox.showwarning("Sin datos", "Carga o pega los datos primero.")
            return
        num = self.df.select_dtypes(include=[np.number])
        descr = num.describe().T
        descr['rango'] = num.max() - num.min()
        descr['var'] = num.var()
        descr['cv'] = descr['std'] / descr['mean'].replace(0, np.nan)
        descr['asimetria'] = num.skew()
        descr['curtosis'] = num.kurt()
        txt = "Resumen estadístico (variables numéricas):\n\n" + descr.to_string(float_format=lambda x: f"{x:.4f}")
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Texto","*.txt")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt)
            messagebox.showinfo("Exportado", f"Análisis exportado a {path}")
        except Exception as e:
            messagebox.showerror("Error al exportar", str(e))

    def clear_data(self):
        self.df = pd.DataFrame()
        self.refresh_table()

    def refresh_table(self):
        for c in self.table.get_children():
            self.table.delete(c)
        self.table["columns"] = ()
        if self.df.empty:
            self.table["show"] = ""
            self.pl_var1['values'] = []
            self.pl_var2['values'] = []
            return

        cols = list(self.df.columns)
        self.table["columns"] = cols
        self.table["show"] = "headings"
        for c in cols:
            self.table.heading(c, text=c)
            self.table.column(c, width=110, anchor='w')
        for i, row in self.df.iterrows():
            values = [self._fmt_val(row.get(c)) for c in cols]
            self.table.insert("", "end", values=values)
            if i >= 999:
                break
        self.pl_var1['values'] = cols
        self.pl_var2['values'] = cols

    def _fmt_val(self, v):
        if pd.isna(v):
            return ""
        return str(v)

    def show_descriptive(self):
        if self.df.empty:
            messagebox.showwarning("Sin datos", "Carga o pega los datos primero.")
            return
        num = self.df.select_dtypes(include=[np.number])
        if num.empty:
            messagebox.showinfo("Sin numéricas", "No hay variables numéricas.")
            return
        descr = num.describe().T
        descr['rango'] = num.max() - num.min()
        descr['var'] = num.var()
        descr['cv'] = descr['std'] / descr['mean'].replace(0, np.nan)
        descr['asimetria'] = num.skew()
        descr['curtosis'] = num.kurt()
        text = descr.to_string(float_format=lambda x: f"{x:.4f}")
        self._show_text_window("Descriptivos (numéricas)", text)

    def show_frequencies(self):
        if self.df.empty:
            messagebox.showwarning("Sin datos", "Carga o pega los datos primero.")
            return
        cats = self.df.select_dtypes(exclude=[np.number])
        if cats.empty:
            messagebox.showinfo("Sin categóricas", "No hay variables categóricas detectadas.")
            return
        out = []
        for c in cats.columns:
            vc = self.df[c].value_counts(dropna=False)
            out.append(f"--- {c} ---\n{vc.to_string()}\n")
        self._show_text_window("Tablas de frecuencia", "\n".join(out))

    def show_correlation(self):
        if self.df.empty:
            messagebox.showwarning("Sin datos", "Carga o pega los datos primero.")
            return
        num = self.df.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            messagebox.showinfo("Pocas variables", "Necesitas al menos 2 variables numéricas para correlación.")
            return
        pear = num.corr(method='pearson')
        spear = num.corr(method='spearman')
        txt = "Correlación Pearson:\n" + pear.to_string(float_format=lambda x: f"{x:.4f}") + "\n\n"
        txt += "Correlación Spearman:\n" + spear.to_string(float_format=lambda x: f"{x:.4f}")
        self._show_text_window("Correlaciones", txt)

    def plot_corr_matrix(self):
        if self.df.empty:
            messagebox.showwarning("Sin datos", "Carga o pega los datos primero.")
            return
        num = self.df.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            messagebox.showinfo("Pocas variables", "Se requieren al menos 2 variables numéricas.")
            return
        fig = plt.Figure()
        ax = fig.add_subplot(111)
        c = num.corr()
        cax = ax.matshow(c)
        fig.colorbar(cax)
        ax.set_xticks(range(len(c.columns)))
        ax.set_yticks(range(len(c.columns)))
        ax.set_xticklabels(c.columns, rotation=90)
        ax.set_yticklabels(c.columns)
        ax.set_title("Matriz de Correlación")
        self._show_figure(fig)

    def regression_dialog(self):
        if self.df.empty:
            messagebox.showwarning("Sin datos", "Carga o pega los datos primero.")
            return
        cols = list(self.df.columns)
        dialog = tk.Toplevel(self.root)
        dialog.title("Regresión: seleccionar variables")
        ttk.Label(dialog, text="Variable dependiente (Y)").pack(anchor=tk.W, padx=6, pady=2)
        ycb = ttk.Combobox(dialog, values=cols); ycb.pack(fill=tk.X, padx=6)
        ttk.Label(dialog, text="Variables independientes (X), separadas por comas").pack(anchor=tk.W, padx=6, pady=2)
        xentry = ttk.Entry(dialog); xentry.pack(fill=tk.X, padx=6, pady=2)
        def runreg():
            ycol = ycb.get().strip()
            xcols = [c.strip() for c in xentry.get().split(',') if c.strip()]
            dialog.destroy()
            if not ycol or not xcols:
                messagebox.showwarning("Faltan variables", "Selecciona Y y al menos una X.")
                return
            self.run_regression(ycol, xcols)
        ttk.Button(dialog, text="Ejecutar", command=runreg).pack(pady=6)

    def run_regression(self, ycol, xcols):
        try:
            df2 = self.df[[ycol] + xcols].dropna()
            Y = df2[ycol]
            X = df2[xcols]
            X = sm.add_constant(X)  
            model = sm.OLS(Y, X).fit()
            self._show_text_window("Regresión - Resumen", model.summary().as_text())
        except Exception as e:
            messagebox.showerror("Error regresión", str(e))

    def tests_dialog(self):
        if self.df.empty:
            messagebox.showwarning("Sin datos", "Carga o pega los datos primero.")
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("Pruebas estadísticas")
        ttk.Label(dialog, text="t-test (2 muestras, variable numérica y variable binaria)").pack(anchor=tk.W, padx=6, pady=2)
        ttk.Label(dialog, text="Variable numérica:").pack(anchor=tk.W, padx=6)
        numcb = ttk.Combobox(dialog, values=list(self.df.select_dtypes(include=[np.number]).columns))
        numcb.pack(fill=tk.X, padx=6)
        ttk.Label(dialog, text="Variable grupo (valores de 2 categorías):").pack(anchor=tk.W, padx=6)
        grcb = ttk.Combobox(dialog, values=list(self.df.select_dtypes(exclude=[np.number]).columns))
        grcb.pack(fill=tk.X, padx=6)

        ttk.Label(dialog, text="").pack()
        ttk.Label(dialog, text="Chi-cuadrado (2 variables categóricas)").pack(anchor=tk.W, padx=6)
        ch1 = ttk.Combobox(dialog, values=list(self.df.columns)); ch1.pack(fill=tk.X, padx=6)
        ch2 = ttk.Combobox(dialog, values=list(self.df.columns)); ch2.pack(fill=tk.X, padx=6)

        ttk.Label(dialog, text="").pack()
        ttk.Label(dialog, text="ANOVA (variable numérica ~ grupo categórico)").pack(anchor=tk.W, padx=6)
        an_num = ttk.Combobox(dialog, values=list(self.df.select_dtypes(include=[np.number]).columns)); an_num.pack(fill=tk.X, padx=6)
        an_grp = ttk.Combobox(dialog, values=list(self.df.select_dtypes(exclude=[np.number]).columns)); an_grp.pack(fill=tk.X, padx=6)

        def run_tests():
            try:
                numcol = numcb.get(); grpcol = grcb.get()
                out = ""
                if numcol and grpcol:
                    g = self.df[[numcol, grpcol]].dropna()
                    vals = g[numcol]; groups = g[grpcol].unique()
                    if len(groups) == 2:
                        a = g[g[grpcol]==groups[0]][numcol]
                        b = g[g[grpcol]==groups[1]][numcol]
                        tstat, pval = stats.ttest_ind(a, b, equal_var=False)
                        out += f"t-test (independientes) {numcol} by {grpcol} ({groups[0]} vs {groups[1]}): t={tstat:.4f}, p={pval:.6f}\n\n"
                    else:
                        out += f"t-test: la variable de grupo no tiene exactamente 2 categorías (tiene {len(groups)})\n\n"
                
                v1 = ch1.get(); v2 = ch2.get()
                if v1 and v2:
                    ct = pd.crosstab(self.df[v1], self.df[v2])
                    chi2, p, dof, ex = stats.chi2_contingency(ct)
                    out += f"Chi2 entre {v1} y {v2}: chi2={chi2:.4f}, p={p:.6f}, dof={dof}\n\n"

                nv = an_num.get(); gv = an_grp.get()
                if nv and gv:
                    gdf = self.df[[nv,gv]].dropna()
                    groups = [gdf[gdf[gv]==lv][nv] for lv in gdf[gv].unique()]
                    if len(groups) >= 2:
                        fstat, pval = stats.f_oneway(*groups)
                        out += f"ANOVA {nv} ~ {gv}: F={fstat:.4f}, p={pval:.6f}\n\n"
                    else:
                        out += "ANOVA: se requieren al menos 2 grupos.\n\n"
                if not out:
                    out = "No se han seleccionado pruebas."
                self._show_text_window("Resultados de pruebas", out)
            except Exception as e:
                messagebox.showerror("Error pruebas", str(e))

        ttk.Button(dialog, text="Ejecutar pruebas", command=run_tests).pack(pady=8)

    def plot_hist(self):
        var = self.pl_var1.get()
        if not var or self.df.empty:
            messagebox.showwarning("Selecciona variable", "Selecciona una variable numérica para el histograma.")
            return
        series = pd.to_numeric(self.df[var], errors='coerce').dropna()
        if series.empty:
            messagebox.showerror("No numérica", "La variable no tiene valores numéricos.")
            return
        fig = plt.Figure()
        ax = fig.add_subplot(111)
        ax.hist(series, bins=12)
        ax.set_title(f"Histograma: {var}")
        ax.set_xlabel(var)
        ax.set_ylabel("Frecuencia")
        self._show_figure(fig)

    def plot_box(self):
        var = self.pl_var1.get()
        if not var or self.df.empty:
            messagebox.showwarning("Selecciona variable", "Selecciona una variable numérica para el boxplot.")
            return
        series = pd.to_numeric(self.df[var], errors='coerce').dropna()
        fig = plt.Figure()
        ax = fig.add_subplot(111)
        ax.boxplot(series, vert=True)
        ax.set_title(f"Boxplot: {var}")
        self._show_figure(fig)

    def plot_scatter(self):
        x = self.pl_var1.get(); y = self.pl_var2.get()
        if not x or not y or self.df.empty:
            messagebox.showwarning("Selecciona X e Y", "Selecciona variables X e Y para el scatter.")
            return
        xser = pd.to_numeric(self.df[x], errors='coerce')
        yser = pd.to_numeric(self.df[y], errors='coerce')
        df2 = pd.DataFrame({x: xser, y: yser}).dropna()
        if df2.empty:
            messagebox.showerror("No numéricas", "Las variables deben ser numéricas.")
            return
        fig = plt.Figure()
        ax = fig.add_subplot(111)
        ax.scatter(df2[x], df2[y])
        ax.set_xlabel(x); ax.set_ylabel(y)
        ax.set_title(f"{y} vs {x}")
        try:
            m, b = np.polyfit(df2[x], df2[y], 1)
            ax.plot(df2[x], m*df2[x] + b, linestyle='--')
        except Exception:
            pass
        self._show_figure(fig)

    def plot_bar(self):
        var = self.pl_var1.get()
        if not var or self.df.empty:
            messagebox.showwarning("Selecciona variable", "Selecciona variable categórica.")
            return
        counts = self.df[var].value_counts()
        fig = plt.Figure()
        ax = fig.add_subplot(111)
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_title(f"Gráfico de barras: {var}")
        ax.set_xlabel(var)
        ax.set_ylabel("Frecuencia")
        self._show_figure(fig)

    def plot_pie(self):
        var = self.pl_var1.get()
        if not var or self.df.empty:
            messagebox.showwarning("Selecciona variable", "Selecciona variable categórica.")
            return
        counts = self.df[var].value_counts()
        fig = plt.Figure()
        ax = fig.add_subplot(111)
        ax.pie(counts.values, labels=counts.index.astype(str), autopct='%1.1f%%')
        ax.set_title(f"Gráfico Circular: {var}")
        self._show_figure(fig)

    def plot_binomial_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Binomial - parámetros")
        ttk.Label(dialog, text="n (ensayos, entero)").pack(anchor=tk.W, padx=6, pady=2)
        nentry = ttk.Entry(dialog); nentry.pack(fill=tk.X, padx=6)
        ttk.Label(dialog, text="p (probabilidad de éxito, 0-1)").pack(anchor=tk.W, padx=6, pady=2)
        pentry = ttk.Entry(dialog); pentry.pack(fill=tk.X, padx=6)
        def run():
            try:
                n = int(nentry.get()); p = float(pentry.get())
                dialog.destroy()
                self.plot_binomial(n, p)
            except Exception as e:
                messagebox.showerror("Parámetros inválidos", str(e))
        ttk.Button(dialog, text="Graficar", command=run).pack(pady=6)

    def plot_binomial(self, n, p):
        x = np.arange(0, n+1)
        y = binom.pmf(x, n, p)
        fig = plt.Figure()
        ax = fig.add_subplot(111)
        ax.bar(x, y)
        ax.set_title(f"Distribución Binomial (n={n}, p={p})")
        ax.set_xlabel("Éxitos")
        ax.set_ylabel("Probabilidad")
        self._show_figure(fig)

    def plot_poisson_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Poisson - parámetro")
        ttk.Label(dialog, text="λ (lambda, media esperada)").pack(anchor=tk.W, padx=6, pady=2)
        lentry = ttk.Entry(dialog); lentry.pack(fill=tk.X, padx=6)
        def run():
            try:
                lam = float(lentry.get())
                dialog.destroy()
                self.plot_poisson(lam)
            except Exception as e:
                messagebox.showerror("Parámetro inválido", str(e))
        ttk.Button(dialog, text="Graficar", command=run).pack(pady=6)

    def plot_poisson(self, lam):
        max_x = max(10, int(lam*4))
        x = np.arange(0, max_x+1)
        y = poisson.pmf(x, lam)
        fig = plt.Figure()
        ax = fig.add_subplot(111)
        ax.bar(x, y)
        ax.set_title(f"Distribución Poisson (λ={lam})")
        ax.set_xlabel("Eventos")
        ax.set_ylabel("Probabilidad")
        self._show_figure(fig)

    def plot_normal_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Normal - parámetros")
        ttk.Label(dialog, text="Usar parámetros de la variable (si es numérica) o introducir mean y sd").pack(anchor=tk.W, padx=6, pady=2)
        ttk.Label(dialog, text="Variable numérica (opcional):").pack(anchor=tk.W, padx=6)
        varcb = ttk.Combobox(dialog, values=list(self.df.select_dtypes(include=[np.number]).columns))
        varcb.pack(fill=tk.X, padx=6)
        ttk.Label(dialog, text="o Mean:").pack(anchor=tk.W, padx=6)
        mean_e = ttk.Entry(dialog); mean_e.pack(fill=tk.X, padx=6)
        ttk.Label(dialog, text="SD:").pack(anchor=tk.W, padx=6)
        sd_e = ttk.Entry(dialog); sd_e.pack(fill=tk.X, padx=6)
        ttk.Label(dialog, text="Sombrear intervalo [a, b] (ej. 20,30)").pack(anchor=tk.W, padx=6)
        range_e = ttk.Entry(dialog); range_e.pack(fill=tk.X, padx=6)
        def run():
            try:
                var = varcb.get().strip()
                if var:
                    ser = pd.to_numeric(self.df[var], errors='coerce').dropna()
                    mean = ser.mean(); sd = ser.std()
                else:
                    mean = float(mean_e.get()); sd = float(sd_e.get())
                a,b = [float(x.strip()) for x in range_e.get().split(',')]
                dialog.destroy()
                self.plot_normal_range(mean, sd, a, b)
            except Exception as e:
                messagebox.showerror("Parámetros inválidos", str(e))
        ttk.Button(dialog, text="Graficar", command=run).pack(pady=6)

    def plot_normal_range(self, mean, sd, a, b):
        x = np.linspace(mean - 4*sd, mean + 4*sd, 500)
        y = norm.pdf(x, mean, sd)
        fig = plt.Figure()
        ax = fig.add_subplot(111)
        ax.plot(x, y)
        xs = np.linspace(a, b, 300)
        ax.fill_between(xs, norm.pdf(xs, mean, sd), alpha=0.4)
        ax.set_title(f"Normal N({mean:.2f},{sd:.2f}) con área [{a},{b}]")
        self._show_figure(fig)

    def save_current_fig(self):
        if self.fig is None:
            messagebox.showwarning("Sin figura", "No hay figura para guardar.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png"),("All","*.*")])
        if not path:
            return
        self.fig.savefig(path)
        messagebox.showinfo("Guardado", f"Gráfico guardado en {path}")

    def _show_figure(self, fig):
        plt.close('all')
        self.fig = fig
        win = tk.Toplevel(self.root)
        win.title("Gráfico")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        ttk.Button(win, text="Cerrar", command=win.destroy).pack(pady=4)

    def _show_text_window(self, title, text):
        win = tk.Toplevel(self.root)
        win.title(title)
        txt = tk.Text(win, wrap='none', width=120, height=30)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert('1.0', text)
        xscroll = ttk.Scrollbar(win, orient='horizontal', command=txt.xview)
        xscroll.pack(side=tk.BOTTOM, fill=tk.X)
        yscroll = ttk.Scrollbar(win, orient='vertical', command=txt.yview)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        txt.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)

if __name__ == "__main__":
    root = tk.Tk()
    app = EstadisticaApp(root)
    root.geometry("1100x750")
    root.mainloop()
