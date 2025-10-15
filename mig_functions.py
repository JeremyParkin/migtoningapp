def standard_sidebar():
    import streamlit as st
    st.sidebar.image('https://app.agilitypr.com/app/assets/images/agility-logo-vertical.png', width=180)
    st.sidebar.subheader('MIG Toning App')
    st.sidebar.caption("Version: October 2025")

    # CSS to adjust sidebar
    adjust_nav = """
                            <style>
                            .eczjsme9, .st-emotion-cache-1wqrzgl {
                                overflow: visible !important;
                                max-width: 250px !important;
                                }
                            .st-emotion-cache-a8w3f8 {
                                overflow: visible !important;
                                }
                            .st-emotion-cache-1cypcdb {
                                max-width: 250px !important;
                                }
                            .e1wa958q1 {
                                filter: brightness(2000%);
                            }
                            
                            </style>
                            """
    # Inject CSS with Markdown
    st.markdown(adjust_nav, unsafe_allow_html=True)

    # Add link to submit bug reports and feature requests
    st.sidebar.markdown(
        "[App Feedback](https://forms.office.com/Pages/ResponsePage.aspx?id=GvcJkLbBVUumZQrrWC6V07d2jCu79C5FsfEZJPZEfZxUNVlIVDRNNVBQVEgxQVFXNEM5VldUMkpXNS4u)")



def format_number(num):
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f} B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f} M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f} K"
    else:
        return str(num)