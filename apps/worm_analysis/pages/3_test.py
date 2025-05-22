import streamlit as st
from hyperscope.helpers.app_helpers import defer as defer_echo
import matplotlib.pyplot as plt

# Assuming the defer_echo function above is defined or imported

st.title("Decorator-based Deferred Execution")

# --- Block 1 ---
st.header("Block 1: Counter")
block1_decorator, run_block1 = defer_echo(key="counter_block")

# The source code of _counter_logic will be displayed here immediately
@block1_decorator
def _counter_logic():
    st.write("Executing Counter Block...")
    if 'app_counter' not in st.session_state:
        st.session_state.app_counter = 0
    st.session_state.app_counter += 1
    st.metric("App Counter Value:", st.session_state.app_counter)
    if st.session_state.app_counter % 3 == 0:
        st.balloons()

st.divider()

# --- Block 2 ---
st.header("Block 2: Text Display")
block2_decorator, run_block2 = defer_echo(key="text_display_block")

# The source code of _text_logic will be displayed here immediately
@block2_decorator
def _text_logic():
    st.subheader("Deferred Text Content")
    st.markdown("This markdown content appears only when `run_block2` is called.")
    st.info("It's a demonstration of deferring different types of content.")

st.divider()

st.header("Block 3: Plots")
block3_decorator, run_block3 = defer_echo(key="mpl_block_3")

# Simple matplotlib example
@block3_decorator
def _mpl_subplots():
    st.write("Executing plotting block with patches...")
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot([1, 3, 2, 4], label="Line") # Some base plot to ensure axes are scaled

    # Add a rectangle patch
    rect = patches.Rectangle((0.5, 0.5), 1.0, 0.8, linewidth=2, edgecolor='r', facecolor='blue', alpha=0.3)
    ax.add_patch(rect)
    ax.set_title("Plot with a Patch")
    ax.legend()


    # Ensure axes limits cover the patch if not covered by other data
    ax.set_xlim(0, 3) # Adjust as needed
    ax.set_ylim(0, 4) # Adjust as needed

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write("Patches block finished.")

# Helper to make session_state JSON serializable

# --- Execution Controls ---
st.header("Controls")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Run Counter Block"):
        run_block1()
with col2:
    if st.button("Run Text Display Block"):
        run_block2()
with col3:
    if st.button("Run plot block"):
        run_block3()


session_state_dict = {}
for k, v in st.session_state.items():
    try:
        # Attempt to convert to string if not a basic type,
        # but be cautious about very complex objects.
        if not isinstance(v, (list, dict, str, int, float, bool, type(None))):
            session_state_dict[k] = str(v)
        else:
            session_state_dict[k] = v
    except Exception:
        session_state_dict[k] = f"COULD_NOT_SERIALIZE_{type(v)}"



st.sidebar.markdown("## Session State Viewer")
st.sidebar.json(session_state_dict)
