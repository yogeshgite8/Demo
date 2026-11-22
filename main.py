import os
import streamlit as st
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END


# -----------------------------
# Streamlit page configuration
# -----------------------------
st.set_page_config(
    page_title="AI HR Recruitment Agent",
    page_icon="🧑‍💼",
    layout="centered"
)

st.title("🧑‍💼 AI HR Recruitment Agent")
st.write(
    "Paste a candidate's application or resume details below. "
    "The agent will assess experience, evaluate Python Developer skills, "
    "and recommend the next HR action."
)


# -----------------------------
# API key and LLM setup
# -----------------------------
def get_openai_api_key():
    """Read the API key from Streamlit Secrets or an environment variable."""
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.getenv("OPENAI_API_KEY")


api_key = get_openai_api_key()

if not api_key:
    st.error(
        "OpenAI API key not found. Add OPENAI_API_KEY to Streamlit Secrets "
        "or set it as an environment variable."
    )
    st.stop()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=api_key
)


# -----------------------------
# LangGraph state
# -----------------------------
class State(TypedDict, total=False):
    application: str
    experience_level: str
    skill_match: str
    response: str


# -----------------------------
# Agent nodes
# -----------------------------
def categorize_experience(state: State) -> dict:
    prompt = ChatPromptTemplate.from_template(
        """Classify the candidate's overall professional experience.
Return exactly ONE of these labels and nothing else:
Entry-level
Mid-level
Senior-level

Candidate application:
{application}"""
    )
    result = (prompt | llm).invoke(
        {"application": state["application"]}
    )
    return {"experience_level": result.content.strip()}


def assess_skillset(state: State) -> dict:
    prompt = ChatPromptTemplate.from_template(
        """Evaluate the candidate for a Python Developer position.
Consider evidence of Python programming and relevant development skills.
Return exactly ONE of these labels and nothing else:
Match
No Match

Candidate application:
{application}"""
    )
    result = (prompt | llm).invoke(
        {"application": state["application"]}
    )
    return {"skill_match": result.content.strip()}


def schedule_hr_interview(state: State) -> dict:
    return {
        "response": "Candidate has been shortlisted for an HR interview."
    }


def escalate_to_recruiter(state: State) -> dict:
    return {
        "response": (
            "Candidate has senior-level experience but doesn't match "
            "the job skills. Escalate to a recruiter for review."
        )
    }


def reject_application(state: State) -> dict:
    return {
        "response": "Candidate doesn't meet the job requirements and has been rejected."
    }


# -----------------------------
# Conditional routing
# -----------------------------
def route_app(state: State) -> str:
    skill = state.get("skill_match", "").strip().lower()
    experience = state.get("experience_level", "").strip().lower()

    # Check "no match" first so it is never mistaken for "match".
    if "no match" not in skill and "match" in skill:
        return "schedule_hr_interview"
    elif "senior" in experience:
        return "escalate_to_recruiter"
    else:
        return "reject_application"


# -----------------------------
# Build and compile the graph
# IMPORTANT: Build it once. Do not add extra edges after compilation.
# -----------------------------
def build_workflow():
    workflow = StateGraph(State)

    workflow.add_node("categorize_experience", categorize_experience)
    workflow.add_node("assess_skillset", assess_skillset)
    workflow.add_node("schedule_hr_interview", schedule_hr_interview)
    workflow.add_node("escalate_to_recruiter", escalate_to_recruiter)
    workflow.add_node("reject_application", reject_application)

    workflow.add_edge(START, "categorize_experience")
    workflow.add_edge("categorize_experience", "assess_skillset")

    workflow.add_conditional_edges(
        "assess_skillset",
        route_app,
        {
            "schedule_hr_interview": "schedule_hr_interview",
            "escalate_to_recruiter": "escalate_to_recruiter",
            "reject_application": "reject_application",
        },
    )

    workflow.add_edge("schedule_hr_interview", END)
    workflow.add_edge("escalate_to_recruiter", END)
    workflow.add_edge("reject_application", END)

    return workflow.compile()


app = build_workflow()


# -----------------------------
# Streamlit user interface
# -----------------------------
with st.form("candidate_screening_form"):
    application_text = st.text_area(
        "Candidate application / resume details",
        height=220,
        placeholder=(
            "Example: I have 5 years of software development experience. "
            "My skills include Python, Django, SQL and REST APIs..."
        ),
    )
    submitted = st.form_submit_button(
        "🔍 Screen Candidate",
        use_container_width=True
    )


if submitted:
    if not application_text.strip():
        st.warning("Please enter the candidate's application details first.")
    else:
        try:
            with st.spinner("AI agent is screening the candidate..."):
                result = app.invoke({"application": application_text.strip()})

            st.subheader("Screening Results")

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Experience Level",
                    result.get("experience_level", "Not available")
                )
            with col2:
                st.metric(
                    "Python Skill Match",
                    result.get("skill_match", "Not available")
                )

            st.markdown("### HR Recommendation")
            recommendation = result.get("response", "No recommendation returned.")

            if "shortlisted" in recommendation.lower():
                st.success(recommendation)
            elif "escalate" in recommendation.lower():
                st.warning(recommendation)
            else:
                st.error(recommendation)

            with st.expander("View submitted application"):
                st.write(application_text.strip())

        except Exception as exc:
            st.error(f"Screening failed: {exc}")
            st.info(
                "Check that your OpenAI API key is valid and that the "
                "required packages are installed."
            )


st.caption(
    "Prototype HR screening assistant. Review AI recommendations before "
    "making employment decisions."
)
