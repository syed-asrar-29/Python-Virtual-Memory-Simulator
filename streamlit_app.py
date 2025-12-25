"""
Virtual Memory Simulator - Streamlit Web Interface
=================================================
Web-based interface for the Virtual Memory Simulator implementing 
page replacement algorithms (FIFO, LRU, Optimal) with interactive visualization.

Author: Virtual Memory Simulator
Date: September 2025
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import plotly.express as px
import plotly.graph_objects as go


class StreamlitPageReplacementSimulator:
    """
    Streamlit-based Virtual Memory Simulator for page replacement algorithms.
    """
    
    def __init__(self, frames, reference_string):
        """
        Initialize the simulator with frame count and reference string.
        
        Args:
            frames (int): Number of memory frames available
            reference_string (list): Sequence of page requests
        """
        self.frames = frames
        self.reference_string = reference_string
        self.results = {}
    
    def fifo_algorithm(self):
        """
        Implements First In First Out (FIFO) page replacement algorithm.
        
        Returns:
            dict: Contains simulation steps, page faults count, and final state
        """
        memory = []
        page_faults = 0
        steps = []
        
        for i, page in enumerate(self.reference_string):
            step = i + 1
            page_fault = False
            
            # Check if page is already in memory
            if page not in memory:
                page_fault = True
                page_faults += 1
                
                # If memory is full, remove the oldest page (FIFO)
                if len(memory) >= self.frames:
                    memory.pop(0)  # Remove first element (oldest)
                
                # Add new page to memory
                memory.append(page)
            
            # Record this step
            memory_state = memory.copy()
            steps.append({
                'Step': step,
                'Page': page,
                'Memory State': ' | '.join(map(str, memory_state)) if memory_state else "Empty",
                'Page Fault': "YES" if page_fault else "NO"
            })
        
        return {
            'algorithm': 'FIFO',
            'steps': steps,
            'page_faults': page_faults,
            'final_memory': memory.copy()
        }
    
    def lru_algorithm(self):
        """
        Implements Least Recently Used (LRU) page replacement algorithm.
        
        Returns:
            dict: Contains simulation steps, page faults count, and final state
        """
        memory = OrderedDict()  # Maintains insertion order for LRU tracking
        page_faults = 0
        steps = []
        
        for i, page in enumerate(self.reference_string):
            step = i + 1
            page_fault = False
            
            if page in memory:
                # Page hit - move to end (most recently used)
                memory.move_to_end(page)
            else:
                # Page fault
                page_fault = True
                page_faults += 1
                
                # If memory is full, remove least recently used page
                if len(memory) >= self.frames:
                    memory.popitem(last=False)  # Remove oldest (LRU)
                
                # Add new page (most recently used)
                memory[page] = True
            
            # Record this step
            memory_list = list(memory.keys())
            steps.append({
                'Step': step,
                'Page': page,
                'Memory State': ' | '.join(map(str, memory_list)) if memory_list else "Empty",
                'Page Fault': "YES" if page_fault else "NO"
            })
        
        return {
            'algorithm': 'LRU',
            'steps': steps,
            'page_faults': page_faults,
            'final_memory': list(memory.keys())
        }
    
    def optimal_algorithm(self):
        """
        Implements Optimal page replacement algorithm (Belady's algorithm).
        
        Returns:
            dict: Contains simulation steps, page faults count, and final state
        """
        memory = []
        page_faults = 0
        steps = []
        
        for i, page in enumerate(self.reference_string):
            step = i + 1
            page_fault = False
            replaced_page = None
            
            # Check if page is already in memory
            if page not in memory:
                page_fault = True
                page_faults += 1
                
                # If memory is full, find optimal page to replace
                if len(memory) >= self.frames:
                    # Find the page that will be used farthest in the future
                    farthest_use = -1
                    page_to_replace = memory[0]  # Default to first page
                    
                    for mem_page in memory:
                        # Find next occurrence of this page in future references
                        next_use = float('inf')  # Assume never used again
                        
                        for j in range(i + 1, len(self.reference_string)):
                            if self.reference_string[j] == mem_page:
                                next_use = j
                                break
                        
                        # Keep track of page with farthest future use
                        if next_use > farthest_use:
                            farthest_use = next_use
                            page_to_replace = mem_page
                    
                    # Replace the optimal page
                    replaced_page = page_to_replace
                    memory.remove(page_to_replace)
                
                # Add new page to memory
                memory.append(page)
            
            # Record this step
            memory_state = memory.copy()
            steps.append({
                'Step': step,
                'Page': page,
                'Memory State': ' | '.join(map(str, memory_state)) if memory_state else "Empty",
                'Page Fault': "YES" if page_fault else "NO",
                'Replaced': str(replaced_page) if replaced_page else "-"
            })
        
        return {
            'algorithm': 'Optimal',
            'steps': steps,
            'page_faults': page_faults,
            'final_memory': memory.copy()
        }
    
    def run_all_algorithms(self):
        """
        Runs all three page replacement algorithms and stores results.
        
        Returns:
            dict: Results from all algorithms
        """
        self.results['FIFO'] = self.fifo_algorithm()
        self.results['LRU'] = self.lru_algorithm()
        self.results['Optimal'] = self.optimal_algorithm()
        
        return self.results


def create_comparison_chart(results):
    """
    Creates an interactive comparison chart using Plotly.
    
    Args:
        results (dict): Algorithm results
        
    Returns:
        plotly figure: Interactive bar chart
    """
    algorithms = list(results.keys())
    page_faults = [results[alg]['page_faults'] for alg in algorithms]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    fig = go.Figure(data=[
        go.Bar(
            x=algorithms,
            y=page_faults,
            marker_color=colors,
            text=page_faults,
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Page Replacement Algorithm Comparison',
        xaxis_title='Algorithm',
        yaxis_title='Number of Page Faults',
        title_font_size=20,
        showlegend=False,
        height=500
    )
    
    return fig


def main():
    """
    Main Streamlit application function.
    """
    # Set page configuration
    st.set_page_config(
        page_title="Virtual Memory Simulator",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main title and description
    st.title("Virtual Memory Simulator")
    st.markdown("""
    ### Page Replacement Algorithm Simulator
    This interactive simulator demonstrates three classic page replacement algorithms:
    - **FIFO** (First In First Out) - Simple queue-based approach
    - **LRU** (Least Recently Used) - Smart tracking of page usage  
    - **Optimal** (Belady's Algorithm) - Theoretical best performance
    """)
    
    # Sidebar for input parameters
    st.sidebar.header("Simulation Parameters")
    
    # Input for number of frames
    frames = st.sidebar.number_input(
        "Number of Memory Frames:",
        min_value=1,
        max_value=10,
        value=3,
        help="The number of memory slots available for pages"
    )
    
    # Choice between demo data and custom input
    input_method = st.sidebar.radio(
        "Choose Input Method:",
        ["Demo Data", "Custom Input"]
    )
    
    if input_method == "Demo Data":
        reference_string = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]
        st.sidebar.success(f"Using demo reference string: {reference_string}")
    else:
        # Custom reference string input
        reference_input = st.sidebar.text_input(
            "Reference String (space-separated):",
            value="7 0 1 2 0 3 0 4 2 3 0 3 2 1 2 0 1 7 0 1",
            help="Enter page numbers separated by spaces"
        )
        
        try:
            reference_string = [int(x) for x in reference_input.split()]
            if len(reference_string) == 0:
                st.sidebar.error("Reference string cannot be empty!")
                return
        except ValueError:
            st.sidebar.error("Please enter valid integers separated by spaces!")
            return
    
    # Display current parameters
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Current Configuration:**")
    st.sidebar.write(f"Frames: {frames}")
    st.sidebar.write(f"Reference String: {reference_string}")
    st.sidebar.write(f"Total Pages: {len(reference_string)}")
    
    # Run simulation button
    if st.sidebar.button("Run Simulation", type="primary"):
        with st.spinner("Running algorithms..."):
            # Create simulator instance
            simulator = StreamlitPageReplacementSimulator(frames, reference_string)
            
            # Run all algorithms
            results = simulator.run_all_algorithms()
            
            # Store results in session state
            st.session_state.results = results
            st.session_state.frames = frames
            st.session_state.reference_string = reference_string
    
    # Display results if available
    if 'results' in st.session_state:
        results = st.session_state.results
        frames = st.session_state.frames
        reference_string = st.session_state.reference_string
        
        # Create tabs for each algorithm
        tab1, tab2, tab3, tab4 = st.tabs(["FIFO", "LRU", "Optimal", "Comparison"])
        
        # FIFO Tab
        with tab1:
            st.header("FIFO Algorithm (First In First Out)")
            fifo_result = results['FIFO']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Page Faults", fifo_result['page_faults'])
            with col2:
                hit_rate = ((len(reference_string) - fifo_result['page_faults']) / len(reference_string)) * 100
                st.metric("Hit Rate", f"{hit_rate:.1f}%")
            with col3:
                st.metric("Fault Rate", f"{(fifo_result['page_faults']/len(reference_string)*100):.1f}%")
            
            st.subheader("Step-by-Step Simulation")
            fifo_df = pd.DataFrame(fifo_result['steps'])
            st.dataframe(fifo_df, width='stretch')
        
        # LRU Tab
        with tab2:
            st.header("LRU Algorithm (Least Recently Used)")
            lru_result = results['LRU']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Page Faults", lru_result['page_faults'])
            with col2:
                hit_rate = ((len(reference_string) - lru_result['page_faults']) / len(reference_string)) * 100
                st.metric("Hit Rate", f"{hit_rate:.1f}%")
            with col3:
                st.metric("Fault Rate", f"{(lru_result['page_faults']/len(reference_string)*100):.1f}%")
            
            st.subheader("Step-by-Step Simulation")
            lru_df = pd.DataFrame(lru_result['steps'])
            st.dataframe(lru_df, width='stretch')
        
        # Optimal Tab
        with tab3:
            st.header("Optimal Algorithm (Belady's)")
            optimal_result = results['Optimal']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Page Faults", optimal_result['page_faults'])
            with col2:
                hit_rate = ((len(reference_string) - optimal_result['page_faults']) / len(reference_string)) * 100
                st.metric("Hit Rate", f"{hit_rate:.1f}%")
            with col3:
                st.metric("Fault Rate", f"{(optimal_result['page_faults']/len(reference_string)*100):.1f}%")
            
            st.subheader("Step-by-Step Simulation")
            optimal_df = pd.DataFrame(optimal_result['steps'])
            st.dataframe(optimal_df, width='stretch')
        
        # Comparison Tab
        with tab4:
            st.header("Algorithm Comparison")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            algorithms = ['FIFO', 'LRU', 'Optimal']
            for i, alg in enumerate(algorithms):
                with [col1, col2, col3][i]:
                    page_faults = results[alg]['page_faults']
                    hit_rate = ((len(reference_string) - page_faults) / len(reference_string)) * 100
                    
                    st.metric(
                        f"{alg} Algorithm",
                        f"{page_faults} faults",
                        f"{hit_rate:.1f}% hit rate"
                    )
            
            # Comparison table
            st.subheader("Detailed Comparison")
            comparison_data = []
            total_requests = len(reference_string)
            
            for alg_name, result in results.items():
                page_faults = result['page_faults']
                hit_rate = ((total_requests - page_faults) / total_requests) * 100
                fault_rate = (page_faults / total_requests) * 100
                
                comparison_data.append({
                    'Algorithm': alg_name,
                    'Page Faults': page_faults,
                    'Hit Rate (%)': f"{hit_rate:.2f}",
                    'Fault Rate (%)': f"{fault_rate:.2f}",
                    'Final Memory State': ' | '.join(map(str, result['final_memory'])) if result['final_memory'] else "Empty"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, width='stretch')
            
            # Interactive comparison chart
            st.subheader("Performance Comparison Chart")
            fig = create_comparison_chart(results)
            st.plotly_chart(fig, width='stretch')
            
            # Best algorithm
            best_algorithm = min(results.items(), key=lambda x: x[1]['page_faults'])
            st.success(f"**Best Performing Algorithm:** {best_algorithm[0]} with {best_algorithm[1]['page_faults']} page faults")
            
    
    else:
        # Instructions when no simulation has been run
        st.info("Configure your simulation parameters in the sidebar and click 'Run Simulation' to see the results!")


if __name__ == "__main__":
    main()