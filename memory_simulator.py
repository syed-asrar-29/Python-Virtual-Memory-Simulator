"""
Virtual Memory Simulator
========================
This project simulates operating system memory management by implementing 
page replacement algorithms (FIFO, LRU, Optimal) and visualizing results.

Author: Virtual Memory Simulator
Date: September 2025
"""

from collections import OrderedDict
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import sys


class PageReplacementSimulator:
    """
    Main class for simulating page replacement algorithms.
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
        
        print("\n" + "="*50)
        print("FIFO ALGORITHM SIMULATION")
        print("="*50)
        
        # Create table for step-by-step output
        table = PrettyTable()
        table.field_names = ["Step", "Page", "Memory State", "Page Fault"]
        
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
                'step': step,
                'page': page,
                'memory': memory_state.copy(),
                'page_fault': page_fault
            })
            
            # Add row to table
            table.add_row([
                step,
                page,
                ' | '.join(map(str, memory_state)) if memory_state else "Empty",
                "YES" if page_fault else "NO"
            ])
        
        print(table)
        print(f"\nTotal Page Faults (FIFO): {page_faults}")
        print(f"Total Pages Accessed: {len(self.reference_string)}")
        print(f"Page Fault Rate: {page_faults/len(self.reference_string)*100:.2f}%")
        
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
        
        print("\n" + "="*50)
        print("LRU ALGORITHM SIMULATION")
        print("="*50)
        
        # Create table for step-by-step output
        table = PrettyTable()
        table.field_names = ["Step", "Page", "Memory State", "Page Fault"]
        
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
                'step': step,
                'page': page,
                'memory': memory_list.copy(),
                'page_fault': page_fault
            })
            
            # Add row to table
            table.add_row([
                step,
                page,
                ' | '.join(map(str, memory_list)) if memory_list else "Empty",
                "YES" if page_fault else "NO"
            ])
        
        print(table)
        print(f"\nTotal Page Faults (LRU): {page_faults}")
        print(f"Total Pages Accessed: {len(self.reference_string)}")
        print(f"Page Fault Rate: {page_faults/len(self.reference_string)*100:.2f}%")
        
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
        
        print("\n" + "="*50)
        print("OPTIMAL ALGORITHM SIMULATION")
        print("="*50)
        
        # Create table for step-by-step output
        table = PrettyTable()
        table.field_names = ["Step", "Page", "Memory State", "Page Fault", "Replaced"]
        
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
                'step': step,
                'page': page,
                'memory': memory_state.copy(),
                'page_fault': page_fault,
                'replaced': replaced_page
            })
            
            # Add row to table
            table.add_row([
                step,
                page,
                ' | '.join(map(str, memory_state)) if memory_state else "Empty",
                "YES" if page_fault else "NO",
                str(replaced_page) if replaced_page else "-"
            ])
        
        print(table)
        print(f"\nTotal Page Faults (Optimal): {page_faults}")
        print(f"Total Pages Accessed: {len(self.reference_string)}")
        print(f"Page Fault Rate: {page_faults/len(self.reference_string)*100:.2f}%")
        
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
        print(f"Running simulation with {self.frames} frames")
        print(f"Reference String: {' -> '.join(map(str, self.reference_string))}")
        
        # Run all algorithms
        self.results['FIFO'] = self.fifo_algorithm()
        self.results['LRU'] = self.lru_algorithm()
        self.results['Optimal'] = self.optimal_algorithm()
        
        return self.results
    
    def display_comparative_analysis(self):
        """
        Displays comparative analysis of all algorithms.
        """
        if not self.results:
            print("No results to compare. Run algorithms first.")
            return
        
        print("\n" + "="*60)
        print("COMPARATIVE ANALYSIS")
        print("="*60)
        
        # Create comparison table
        comparison_table = PrettyTable()
        comparison_table.field_names = ["Algorithm", "Page Faults", "Hit Rate", "Efficiency"]
        
        total_requests = len(self.reference_string)
        
        for alg_name, result in self.results.items():
            page_faults = result['page_faults']
            hit_rate = ((total_requests - page_faults) / total_requests) * 100
            efficiency = f"{hit_rate:.2f}%"
            
            comparison_table.add_row([
                alg_name,
                page_faults,
                f"{hit_rate:.2f}%",
                efficiency
            ])
        
        print(comparison_table)
        
        # Find best algorithm
        best_algorithm = min(self.results.items(), key=lambda x: x[1]['page_faults'])
        print(f"\nBest Performing Algorithm: {best_algorithm[0]} with {best_algorithm[1]['page_faults']} page faults")
    
    def plot_comparison_chart(self):
        """
        Creates a bar chart comparing page faults across algorithms.
        """
        if not self.results:
            print("No results to plot. Run algorithms first.")
            return
        
        algorithms = list(self.results.keys())
        page_faults = [self.results[alg]['page_faults'] for alg in algorithms]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(algorithms, page_faults, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        
        # Add value labels on bars
        for bar, fault_count in zip(bars, page_faults):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(fault_count), ha='center', va='bottom', fontweight='bold')
        
        plt.title('Page Replacement Algorithm Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Algorithm', fontsize=12)
        plt.ylabel('Number of Page Faults', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Add reference string info
        plt.figtext(0.02, 0.02, f"Reference String: {self.reference_string}\nFrames: {self.frames}", 
                   fontsize=8, style='italic')
        
        plt.tight_layout()
        
        # Save chart as image file
        chart_filename = 'page_replacement_comparison.png'
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print(f"\nComparison chart saved as '{chart_filename}'")
        print("Chart shows the number of page faults for each algorithm.")


class InputHandler:
    """
    Handles user input for the simulation parameters.
    """
    
    @staticmethod
    def get_frames():
        """
        Gets the number of frames from user input.
        
        Returns:
            int: Number of frames
        """
        while True:
            try:
                frames = int(input("\nEnter the number of frames (memory slots): "))
                if frames <= 0:
                    print("Number of frames must be positive. Please try again.")
                    continue
                return frames
            except ValueError:
                print("Please enter a valid integer.")
    
    @staticmethod
    def get_reference_string():
        """
        Gets the reference string from user input.
        
        Returns:
            list: List of page numbers
        """
        while True:
            try:
                print("\nEnter the reference string (page numbers separated by spaces):")
                print("Example: 7 0 1 2 0 3 0 4 2 3 0 3 2 1 2 0 1 7 0 1")
                
                reference_input = input("Reference string: ").strip()
                if not reference_input:
                    print("Reference string cannot be empty. Please try again.")
                    continue
                
                reference_string = [int(x) for x in reference_input.split()]
                if len(reference_string) == 0:
                    print("Reference string must contain at least one page. Please try again.")
                    continue
                
                return reference_string
            except ValueError:
                print("Please enter valid integers separated by spaces.")
    
    @staticmethod
    def get_demo_data():
        """
        Provides demo data for quick testing.
        
        Returns:
            tuple: (frames, reference_string)
        """
        frames = 3
        reference_string = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]
        
        print(f"\nUsing demo data:")
        print(f"Frames: {frames}")
        print(f"Reference String: {reference_string}")
        
        return frames, reference_string


def main():
    """
    Main function to run the Virtual Memory Simulator.
    """
    print("="*60)
    print("        VIRTUAL MEMORY SIMULATOR")
    print("     Page Replacement Algorithm Simulator")
    print("="*60)
    print("\nThis simulator implements three page replacement algorithms:")
    print("1. FIFO (First In First Out)")
    print("2. LRU (Least Recently Used)")
    print("3. Optimal (Belady's Algorithm)")
    print("\nFeatures:")
    print("- Step-by-step simulation display")
    print("- Page fault counting and analysis")
    print("- Comparative performance analysis")
    print("- Visualization with charts")
    
    # Get user choice for input method
    while True:
        print("\nChoose input method:")
        print("1. Enter custom data")
        print("2. Use demo data")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            # Get custom input
            frames = InputHandler.get_frames()
            reference_string = InputHandler.get_reference_string()
            break
        elif choice == '2':
            # Use demo data
            frames, reference_string = InputHandler.get_demo_data()
            break
        elif choice == '3':
            print("Exiting simulator. Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    # Create simulator instance
    simulator = PageReplacementSimulator(frames, reference_string)
    
    # Run all algorithms
    results = simulator.run_all_algorithms()
    
    # Display comparative analysis
    simulator.display_comparative_analysis()
    
    # Plot comparison chart
    print("\nGenerating comparison chart...")
    simulator.plot_comparison_chart()
    
    print("\n" + "="*60)
    print("Simulation completed successfully!")
    print("Thank you for using the Virtual Memory Simulator!")
    print("="*60)


if __name__ == "__main__":
    main()