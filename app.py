import os, sqlite3, tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re  # for email validation

# --- Database setup ---
BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "shop3.db")

def db():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = db(); c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS customers(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL, 
        email TEXT NOT NULL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS products(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL, 
        price REAL NOT NULL CHECK(price>=0))""")
    c.execute("""CREATE TABLE IF NOT EXISTS orders(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER NOT NULL,
        product_id  INTEGER NOT NULL,
        quantity INTEGER NOT NULL CHECK(quantity>0),
        unit_price REAL NOT NULL CHECK(unit_price>=0),
        order_date TEXT NOT NULL DEFAULT (date('now')),
        FOREIGN KEY(customer_id) REFERENCES customers(id) ON DELETE CASCADE,
        FOREIGN KEY(product_id)  REFERENCES products(id)  ON DELETE CASCADE)""")

    # --- sample data ---
    if c.execute("SELECT COUNT(*) FROM customers").fetchone()[0]==0:
        c.executemany("INSERT INTO customers(name,email) VALUES(?,?)",
                      [("Alice","alice@gmail.com"),("Bob","bob@gmail.com")])
    if c.execute("SELECT COUNT(*) FROM products").fetchone()[0]==0:
        c.executemany("INSERT INTO products(name,price) VALUES(?,?)",
                      [("Coffee 1kg",14.99),("Mug",3.99),("Espresso Machine",199.00)])
    conn.commit(); conn.close()

# --- Validation functions ---
def req_text(v, field):
    v = str(v).strip()
    if not v:
        raise ValueError(f"{field} is required")
    # check for email format
    if "email" in field.lower():
        pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
        if not re.match(pattern, v):
            raise ValueError("Invalid email format. Please include '@' and domain (e.g., user@gmail.com).")
    return v

def req_int(v, field, minv=0):
    try: n=int(str(v).strip())
    except: raise ValueError(f"{field} must be an integer")
    if n<minv: raise ValueError(f"{field} must be ≥ {minv}")
    return n

def req_float(v, field, minv=0.0):
    try: n=float(str(v).strip())
    except: raise ValueError(f"{field} must be a number")
    if n<minv: raise ValueError(f"{field} must be ≥ {minv}")
    return n

# --- Generic Table Window ---
def open_table(title, table, cols, pk, form, related=None):
    w = tk.Toplevel(root); w.title(title); w.geometry("860x520")
    bar = ttk.Frame(w); bar.pack(fill="x", padx=6, pady=6)
    tree = ttk.Treeview(w, columns=cols, show="headings", selectmode="browse")
    for c in cols: tree.heading(c, text=c); tree.column(c, width=130, anchor="center")
    tree.pack(fill="both", expand=True, padx=8, pady=8)

    def refresh():
        tree.delete(*tree.get_children())
        with db() as conn:
            for r in conn.execute(f"SELECT {', '.join(cols)} FROM {table}"):
                tree.insert("", "end", values=r)

    def selected():
        i = tree.focus()
        return dict(zip(cols, tree.item(i,"values"))) if i else None

    def add_edit(mode):
        r = selected()
        if mode=="edit" and not r: return messagebox.showinfo("Info","Select a row")
        f = tk.Toplevel(w); f.title(mode.title()); f.geometry("380x260")
        entries={}
        for i,(field,label,typ) in enumerate(form):
            ttk.Label(f,text=label).grid(row=i,column=0,sticky="e",padx=6,pady=6)
            e=ttk.Entry(f,width=28); e.grid(row=i,column=1,sticky="w"); entries[field]=(e,typ)
            if mode=="edit" and field in (r or {}): e.insert(0, r[field])

        def save():
            try:
                data={}
                for field,(e,typ) in entries.items():
                    raw=e.get()
                    data[field]=(req_text(raw,field) if typ=="text" 
                                 else req_int(raw,field,0) if typ=="int"  
                                 else req_float(raw,field,0.0))
                with db() as conn:
                    c=conn.cursor()
                    if mode=="add":
                        fs=", ".join(data.keys()); qs=", ".join(["?"]*len(data))
                        c.execute(f"INSERT INTO {table}({fs}) VALUES({qs})", tuple(data.values()))
                    else:
                        sets=", ".join([f"{k}=?" for k in data.keys()])
                        c.execute(f"UPDATE {table} SET {sets} WHERE {pk}=?", (*data.values(), r[pk]))
                f.destroy(); refresh()
            except Exception as ex: messagebox.showerror("Validation", str(ex))

        ttk.Button(f,text="Save",command=save).grid(row=len(form),column=0,columnspan=2,pady=8)

    def delete():
        r=selected()
        if not r: return messagebox.showinfo("Info","Select a row")
        if not messagebox.askyesno("Confirm","Delete selected row?"): return
        with db() as conn: conn.execute(f"DELETE FROM {table} WHERE {pk}=?", (r[pk],))
        refresh()

    def import_rows():
        path=filedialog.askopenfilename(title="Import",filetypes=[("CSV","*.csv"),("Excel","*.xlsx *.xls")])
        if not path: return
        try:
            df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
            expected=[f for f,_,_ in form]
            missing=[c for c in expected if c not in df.columns]
            extra=[c for c in df.columns if c not in expected]
            if missing: raise ValueError(f"Missing columns: {missing}")
            if extra:   raise ValueError(f"Unexpected columns: {extra}")
            with db() as conn:
                c=conn.cursor()
                for i,row in df.iterrows():
                    data={}
                    for f,_,t in form:
                        v=row.get(f,"")
                        data[f]=(req_text(v,f) if t=="text" else req_int(v,f,0) if t=="int" else req_float(v,f,0.0))
                    if table=="orders":
                        if not c.execute("SELECT 1 FROM customers WHERE id=?", (data["customer_id"],)).fetchone():
                            raise ValueError(f"Row {i}: customer_id {data['customer_id']} not found")
                        if not c.execute("SELECT 1 FROM products WHERE id=?", (data["product_id"],)).fetchone():
                            raise ValueError(f"Row {i}: product_id {data['product_id']} not found")
                    fs=", ".join(data.keys()); qs=", ".join(["?"]*len(data))
                    c.execute(f"INSERT INTO {table}({fs}) VALUES({qs})", tuple(data.values()))
            refresh(); messagebox.showinfo("Import","Done.")
        except Exception as ex: messagebox.showerror("Import failed", str(ex))

    def export_rows():
        path=filedialog.asksaveasfilename(title="Export",defaultextension=".csv",
              filetypes=[("CSV","*.csv"),("Excel","*.xlsx")])
        if not path: return
        with db() as conn:
            df=pd.read_sql_query(f"SELECT {', '.join(cols)} FROM {table}", conn)
        (df.to_excel(path,index=False) if path.lower().endswith(".xlsx") else df.to_csv(path,index=False))
        messagebox.showinfo("Export","Saved.")

    def related_open(cb):
        r=selected()
        if not r: return messagebox.showinfo("Info","Select a row")
        cb(r)

    ttk.Button(bar,text="Add",command=lambda:add_edit("add")).pack(side="left",padx=3)
    ttk.Button(bar,text="Edit",command=lambda:add_edit("edit")).pack(side="left",padx=3)
    ttk.Button(bar,text="Delete",command=delete).pack(side="left",padx=3)
    ttk.Button(bar,text="Import",command=import_rows).pack(side="left",padx=3)
    ttk.Button(bar,text="Export",command=export_rows).pack(side="left",padx=3)
    for lab,cb in (related or []):
        ttk.Button(bar,text=lab,command=lambda cb=cb:related_open(cb)).pack(side="left",padx=3)

    refresh()

# --- Related popups ---
def popup_df(title, df):
    w=tk.Toplevel(root); w.title(title); w.geometry("760x420")
    cols=list(df.columns) if not df.empty else []
    tv=ttk.Treeview(w, columns=cols, show="headings")
    for c in cols: tv.heading(c,text=c); tv.column(c,width=120,anchor="center")
    tv.pack(fill="both",expand=True,padx=8,pady=8)
    for _,r in df.iterrows(): tv.insert("", "end", values=list(r.values))

def show_orders_of_customer(row):
    with db() as conn:
        df=pd.read_sql_query("SELECT id,customer_id,product_id,quantity,unit_price,order_date FROM orders WHERE customer_id=?",
                             conn, params=(row["id"],))
    popup_df(f"Orders of Customer #{row['id']}", df)

def show_orders_of_product(row):
    with db() as conn:
        df=pd.read_sql_query("SELECT id,customer_id,product_id,quantity,unit_price,order_date FROM orders WHERE product_id=?",
                             conn, params=(row["id"],))
    popup_df(f"Orders of Product #{row['id']}", df)

def open_customer_of_order(row):
    with db() as conn:
        df=pd.read_sql_query("""SELECT c.id,c.name,c.email FROM orders o
                                JOIN customers c ON c.id=o.customer_id WHERE o.id=?""",
                              conn, params=(row["id"],))
    popup_df(f"Customer of Order #{row['id']}", df)

def open_product_of_order(row):
    with db() as conn:
        df=pd.read_sql_query("""SELECT p.id,p.name,p.price FROM orders o
                                JOIN products p ON p.id=o.product_id WHERE o.id=?""",
                              conn, params=(row["id"],))
    popup_df(f"Product of Order #{row['id']}", df)

# --- Reports ---
def open_reports():
    w=tk.Toplevel(root); w.title("Reports"); w.geometry("960x600")
    bar=ttk.Frame(w); bar.pack(fill="x", padx=8, pady=8)
    area=ttk.Frame(w); area.pack(fill="both", expand=True)
    def clear(): [child.destroy() for child in area.winfo_children()]

    def summary():
        clear()
        with db() as conn:
            df=pd.read_sql_query("""SELECT COALESCE(SUM(quantity),0) items,
                                    COALESCE(ROUND(SUM(quantity*unit_price),2),0) revenue
                                    FROM orders""", conn)
        t=tk.Text(area,height=6); t.pack(fill="both",expand=True,padx=8,pady=8)
        t.insert("end", f"Total items sold: {int(df['items'][0])}\n")
        t.insert("end", f"Total revenue   : {float(df['revenue'][0]):.2f}\n")

    def best_sellers():
        clear()
        with db() as conn:
            df=pd.read_sql_query("""SELECT p.name, SUM(o.quantity) qty
                                    FROM orders o JOIN products p ON p.id=o.product_id
                                    GROUP BY p.name ORDER BY qty DESC LIMIT 10""", conn)
        if df.empty: return messagebox.showinfo("Info","No data.")
        fig=Figure(figsize=(7,4)); ax=fig.add_subplot(111)
        df.plot(kind="bar", x="name", y="qty", ax=ax)
        FigureCanvasTkAgg(fig, master=area).get_tk_widget().pack(fill="both",expand=True)

    def top_orders():
        clear()
        with db() as conn:
            df=pd.read_sql_query("""SELECT id AS order_id, ROUND(quantity*unit_price,2) total
                                    FROM orders ORDER BY total DESC LIMIT 10""", conn)
        if df.empty: return messagebox.showinfo("Info","No data.")
        fig=Figure(figsize=(7,4)); ax=fig.add_subplot(111)
        df.plot(kind="bar", x="order_id", y="total", ax=ax)
        FigureCanvasTkAgg(fig, master=area).get_tk_widget().pack(fill="both",expand=True)

    ttk.Button(bar,text="Summary",command=summary).pack(side="left",padx=4)
    ttk.Button(bar,text="Best Sellers",command=best_sellers).pack(side="left",padx=4)
    ttk.Button(bar,text="Top Orders",command=top_orders).pack(side="left",padx=4)

# --- Table openers ---
def open_customers():
    open_table("Customers","customers",
        cols=["id","name","email"], pk="id",
        form=[("name","Name","text"),("email","Email","text")],
        related=[("View Orders of Customer", show_orders_of_customer)])

def open_products():
    open_table("Products","products",
        cols=["id","name","price"], pk="id",
        form=[("name","Name","text"),("price","Price","float")],
        related=[("View Orders of Product", show_orders_of_product)])

def open_orders():
    open_table("Orders","orders",
        cols=["id","customer_id","product_id","quantity","unit_price","order_date"], pk="id",
        form=[("customer_id","Customer ID","int"),
              ("product_id","Product ID","int"),
              ("quantity","Quantity","int"),
              ("unit_price","Unit Price","float")],
        related=[("Open Customer", open_customer_of_order),
                 ("Open Product",  open_product_of_order)])

# --- Main Window ---
init_db()
root=tk.Tk(); root.title("Simple Order Management"); root.geometry("420x360")
ttk.Label(root,text="Welcome!",font=("Segoe UI",16,"bold")).pack(pady=12)
ttk.Button(root,text="Customers",width=28,command=open_customers).pack(pady=5)
ttk.Button(root,text="Products",width=28,command=open_products).pack(pady=5)
ttk.Button(root,text="Orders",width=28,command=open_orders).pack(pady=5)
ttk.Button(root,text="Reports",width=28,command=open_reports).pack(pady=5)
root.mainloop()
