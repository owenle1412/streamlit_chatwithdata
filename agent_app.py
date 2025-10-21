# agent_app.py
import os
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType

# =========================================================
# 🔑 Streamlit Cloud secret management
# (Add OPENAI_API_KEY in Streamlit → Settings → Secrets)
# =========================================================
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     st.error("❌ Missing OpenAI API key. Please set OPENAI_API_KEY in Streamlit Secrets.")
#     st.stop()
OPENAI_API_KEY="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"
# =========================================================
# 🗂️ Paths and setup
# =========================================================
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "sales_qtr.csv")
DB_PATH = os.path.join(DATA_DIR, "sales.db")
os.makedirs(DATA_DIR, exist_ok=True)

# =========================================================
# 📦 Ensure SQLite DB exists (auto-create from CSV)
# =========================================================
@st.cache_data
def ensure_db():
    if not os.path.exists(DB_PATH):
        if not os.path.exists(CSV_PATH):
            st.error("CSV file not found. Please upload or provide data/sales_qtr.csv.")
            st.stop()
        df = pd.read_csv(CSV_PATH)
        engine = create_engine(f"sqlite:///{DB_PATH}")
        df.to_sql("sales", engine, if_exists="replace", index=False)
        st.success("✅ Created SQLite DB from CSV.")
    return f"sqlite:///{DB_PATH}"

SQLALCHEMY_URL = ensure_db()

# =========================================================
# 🧠 LangChain SQL Agent Setup
# =========================================================
@st.cache_resource
def load_agent():
    db = SQLDatabase.from_uri(SQLALCHEMY_URL)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
    )
    return agent, db

agent, db = load_agent()

# =========================================================
# 🎨 Streamlit UI
# =========================================================
st.set_page_config(layout="wide", page_title="AI SQL Agent")
st.title("🤖 AI SQL Agent — Natural Language to Data Insights")
st.caption("Ask questions about your data in plain English. Example: *'Show me total revenue by product in Q3.'*")

# =========================================================
# 🧾 Question input
# =========================================================
query = st.text_input("💬 Ask a question about your sales data:", "Show me total revenue by product in Q3")

if st.button("Run Query"):
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("🧠 Thinking... running query and analyzing results..."):
        try:
            # Run SQL agent
            response = agent.invoke({"input": query})
            answer_text = response["output"]

            st.subheader("📊 AI-Generated Insight")
            st.markdown(answer_text)

            # =========================================================
            # 🧩 Try to extract the actual SQL and show table results
            # =========================================================
            # The agent internally runs SQL, but we can re-run it manually
            # if we want to show the underlying data
            from sqlalchemy import text

            engine = create_engine(SQLALCHEMY_URL)
            # Simple heuristic: if it mentions "SELECT", extract the SQL
            if "SELECT" in answer_text.upper():
                sql_start = answer_text.upper().find("SELECT")
                sql_snippet = answer_text[sql_start:]
                st.code(sql_snippet, language="sql")

                try:
                    df = pd.read_sql(text(sql_snippet), engine)
                    if not df.empty:
                        st.dataframe(df)
                        # Optional: simple chart if columns available
                        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                        if numeric_cols:
                            num_col = numeric_cols[0]
                            group_cols = [c for c in df.columns if df[c].dtype == object]
                            if group_cols:
                                x_col = group_cols[0]
                                agg = df.groupby(x_col)[num_col].sum().reset_index()
                                st.bar_chart(agg.set_index(x_col))
                except Exception:
                    pass

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

# =========================================================
# 📂 Optional: show schema and sample data
# =========================================================
with st.expander("📋 View database schema and sample data"):
    st.markdown("**Tables:** " + ", ".join(db.get_usable_table_names()))
    try:
        engine = create_engine(SQLALCHEMY_URL)
        df = pd.read_sql("SELECT * FROM sales LIMIT 5;", engine)
        st.dataframe(df)
    except Exception as e:
        st.error(f"Could not show sample data: {e}")
