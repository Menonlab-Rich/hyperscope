import streamlit as st
st.set_page_config(
    page_title="Welcome",
    page_icon="👋",
)

'''
    Okay, this sounds like an exciting application! Here's some copy for the landing page of your Streamlit app, aiming to be informative and engaging:

---

## 🚀 EventFlow Tracker: Illuminating Motion from Event Streams

**Welcome to EventFlow Tracker – your interactive window into the world of dynamic vision!**

This application brings to life a cutting-edge pipeline for detecting and tracking objects from event camera data. Witness how sparse, asynchronous events are transformed into meaningful, persistent object trajectories right before your eyes.

---

### ✨ How It Works: From Raw Events to Robust Tracks

Our process intelligently deciphers motion from the unique data provided by event cameras:

1.  **⚡ Event-Driven Sensing:** We start with data from event cameras, which, unlike traditional cameras, only record changes in a scene (pixel-level brightness changes). This means high temporal resolution, low latency, and excellent performance in challenging lighting conditions – perfect for capturing fast-moving objects.
2.  **🧠 Intelligent Clustering (DBSCAN):** In discrete time windows, the incoming stream of (x, y, t) events is analyzed. We use the robust **DBSCAN algorithm** to group spatially and temporally proximal events, identifying potential "detections" or object candidates amidst the noise.
3.  **🎯 Persistent Tracking (SORT):** These fleeting detections are then fed into the **Simple Online and Realtime Tracking (SORT)** algorithm. SORT efficiently associates detections across time windows, assigning consistent IDs to objects and maintaining their tracks even through occlusions or brief disappearances, all powered by Kalman filtering and the Hungarian algorithm.

---

### 🔬 Explore and Discover with EventFlow Tracker!

This interactive platform allows you to:

* **Process Event Data:** Upload your own event camera recordings (RAW or HDF5 format containing events) or explore with our sample datasets to see the tracker in action.
* **Tune and Visualize:** Adjust key parameters for DBSCAN (like `epsilon` and `min_samples`) and SORT (such as `max_age`, `min_hits`, and `IoU threshold`) and observe their impact on clustering and tracking performance.
* **See the Stages:** Visualize the raw event stream, the dynamically formed event clusters (your detections), and the final, color-coded object tracks overlaid on a representation of the scene.
* **Analyze Results:** The generated tracks and their corresponding source event clusters can be saved to an HDF5 file, allowing for detailed offline analysis and integration with your research workflows.

---

### 💡 Get Started!

Ready to see the unseen? Upload your data, tweak the parameters, and watch as EventFlow Tracker brings dynamic scenes to life. Discover the power and potential of event-based vision and advanced tracking algorithms.

---

**Tips for your Streamlit App:**

* Keep this copy on the main landing page or an "About" section.
* Use Streamlit's layout features (columns, expanders) to present information clearly.
* If possible, include a small animation or GIF showing the app in action (events -> clusters -> tracks) to make it even more engaging!
* Clearly label input fields for parameters and provide sensible default values and tooltips.

Good luck with your Streamlit app! It sounds like a very valuable tool.
                                   '''
