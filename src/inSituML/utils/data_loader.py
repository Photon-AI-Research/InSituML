import sys
import os
import numpy as np
import openpmd_api as io
import json
import gc
import tempfile


class DataLoader:
    """
    Data loader class to read simulation data (openPMD files) and produce batches for a 3D convolutional VAE.
    Supports loading vector fields (e.g., B, E) as 3-channel inputs and scalar fields with memory-efficient techniques.
    """

    def __init__(self, root_path: str = "/bigdata/hplsim/production/LWFA_GordonBell_dataLoader/runs/001_LWFA",
                 openPMDPath: str = "simOutput/openPMD",
                 metadata_path: str = "picongpu.json",
                 mesh_names: list = ["B", "E"],
                 batch_size: int = 8,
                 iteration: int = 20000,
                 normalize: bool = True,
                 chunk_size: tuple = (128, 448, 128)):
        """
        Initialize the data loader.
        
        Args:
            root_path (str): Root directory of the simulation data.
            openPMDPath (str): Path to openPMD files relative to root_path.
            metadata_path (str): Path to metadata JSON file.
            mesh_names (list): List of mesh names to load (e.g., ["B", "E", "e_all_chargeDensity"]).
            batch_size (int): Number of samples per batch.
            iteration (int): Simulation iteration to load.
            normalize (bool): Whether to normalize data to [0, 1].
            chunk_size (tuple): Size of chunks (D, H, W) for loading data.
        """
        self.root_path = root_path
        self.openPMDPath = openPMDPath
        self.metadata_path = metadata_path
        self.mesh_names = mesh_names
        self.batch_size = batch_size
        self.iteration = iteration
        self.normalize = normalize
        self.chunk_size = chunk_size
        self.series = None
        self._batch_index = 0
        self.temp_dir = tempfile.mkdtemp()  # Temporary directory for memory-mapped files

        # Set up ADIOS path
        adiosRoot = os.environ.get("ADIOS_ROOT")
        if adiosRoot is None:
            raise EnvironmentError("ADIOS_ROOT environment variable is not set")
        pythonVersion = ".".join(sys.version.split()[0].split('.')[:2])
        sys.path.append(f"{adiosRoot}/lib/python{pythonVersion}/site-packages")

        # Read metadata from picongpu.json
        json_path = os.path.join(self.metadata_path)
        try:
            with open(json_path, 'r') as fp:
                self.metadata = json.load(fp)
        except FileNotFoundError:
            print(f"Error: {json_path} not found")
            sys.exit(1)

        # Initialize series
        self.get_series()

    def get_series(self, pathPattern: str = "simOutput_fields_%T.bp"):
        """Open the openPMD series."""
        self.series_path = os.path.join(self.root_path, self.openPMDPath, pathPattern)
        try:
            self.series = io.Series(self.series_path, io.Access.read_only)
            print(f"[DEBUG] Opened series: {self.series_path}")
        except Exception as e:
            print(f"Error opening series {self.series_path}: {e}")
            sys.exit(1)

    def get_iteration(self, key: int):
        """Access a specific iteration from the series."""
        if self.series is None:
            self.get_series()
        try:
            iteration = self.series.iterations[key]
            print(f"[DEBUG] Accessed iteration: {key}")
            return iteration
        except Exception as e:
            print(f"Error accessing iteration {key}: {e}")
            self.series.close()
            sys.exit(1)

    def load_mesh_data(self, mesh_name: str, iteration: int = 20000):
        """
        Load mesh data in chunks using contiguous temporary buffers.
        For vector fields (e.g., B, E), stack x, y, z components as channels.
        For scalar fields, return a single channel.
        
        Returns:
            np.ndarray: Array of shape (C, D, H, W) where C is 3 for vector fields, 1 for scalars.
        """
        print(f"[DEBUG] Loading mesh: {mesh_name}")
        curr_iteration = self.get_iteration(iteration)
        mesh = curr_iteration.meshes[mesh_name]
        full_shape = (512, 1792, 512)  # Original shape

        if mesh_name in ["B", "E", "J"]:  # Vector fields
            components = ["x", "y", "z"]
            data_list = []
            for comp in components:
                print(f"[DEBUG] Loading component {comp} for mesh {mesh_name}")
                comp_data = mesh[comp]
                # Create memory-mapped array for the component
                temp_file = os.path.join(self.temp_dir, f"{mesh_name}_{comp}.npy")
                arr = np.memmap(temp_file, dtype=np.float32, mode='w+', shape=full_shape)
                print(f"[DEBUG] Created memmap array, shape: {arr.shape}, size: {arr.nbytes / 1e6:.2f} MB")
                
                # Load data in chunks
                for z in range(0, full_shape[0], self.chunk_size[0]):
                    for y in range(0, full_shape[1], self.chunk_size[1]):
                        for x in range(0, full_shape[2], self.chunk_size[2]):
                            # Calculate actual extent for this chunk
                            z_end = min(z + self.chunk_size[0], full_shape[0])
                            y_end = min(y + self.chunk_size[1], full_shape[1])
                            x_end = min(x + self.chunk_size[2], full_shape[2])
                            extent = [z_end - z, y_end - y, x_end - x]
                            
                            # Create contiguous chunk with exact size needed
                            chunk = np.empty(extent, dtype=np.float32, order='C')
                            # print(f"[DEBUG] Allocated chunk, shape: {chunk.shape}, size: {chunk.nbytes / 1e6:.2f} MB, contiguous: {chunk.flags['C_CONTIGUOUS']}")
                            
                            # Load chunk with proper offset and extent
                            comp_data.load_chunk(chunk, offset=[z, y, x], extent=extent)
                            self.series.flush()
                            
                            # Copy to memory-mapped array
                            arr[z:z_end, y:y_end, x:x_end] = chunk
                            # print(f"[DEBUG] Loaded chunk at offset {[z, y, x]}, extent {extent}")
                            del chunk
                            gc.collect()
                data_list.append(arr)

            # Stack components - load from memmap to regular array for stacking
            print(f"[DEBUG] Stacking {len(data_list)} components")
            # Create output array
            data = np.empty((3, *full_shape), dtype=np.float32, order='C')
            for i, component_data in enumerate(data_list):
                # Load memmap data in chunks to avoid memory issues
                for z in range(0, full_shape[0], self.chunk_size[0]):
                    for y in range(0, full_shape[1], self.chunk_size[1]):
                        for x in range(0, full_shape[2], self.chunk_size[2]):
                            z_end = min(z + self.chunk_size[0], full_shape[0])
                            y_end = min(y + self.chunk_size[1], full_shape[1])
                            x_end = min(x + self.chunk_size[2], full_shape[2])
                            data[i, z:z_end, y:y_end, x:x_end] = component_data[z:z_end, y:y_end, x:x_end]
                # Clean up memmap file
                del component_data
            
            print(f"[DEBUG] Stacked data, shape: {data.shape}, size: {data.nbytes / 1e6:.2f} MB")
            del data_list
            gc.collect()
            
        else:  # Scalar fields
            print(f"[DEBUG] Loading scalar component for mesh {mesh_name}")
            comp_data = mesh[io.Mesh_Record_Component.SCALAR]
            
            # Create output array directly
            data = np.empty((1, *full_shape), dtype=np.float32, order='C')
            print(f"[DEBUG] Created output array, shape: {data.shape}, size: {data.nbytes / 1e6:.2f} MB")
            
            # Load data in chunks directly to output array
            for z in range(0, full_shape[0], self.chunk_size[0]):
                for y in range(0, full_shape[1], self.chunk_size[1]):
                    for x in range(0, full_shape[2], self.chunk_size[2]):
                        # Calculate actual extent for this chunk
                        z_end = min(z + self.chunk_size[0], full_shape[0])
                        y_end = min(y + self.chunk_size[1], full_shape[1])
                        x_end = min(x + self.chunk_size[2], full_shape[2])
                        extent = [z_end - z, y_end - y, x_end - x]
                        
                        # Create contiguous chunk with exact size needed
                        chunk = np.empty(extent, dtype=np.float32, order='C')
                        # print(f"[DEBUG] Allocated chunk, shape: {chunk.shape}, size: {chunk.nbytes / 1e6:.2f} MB, contiguous: {chunk.flags['C_CONTIGUOUS']}")
                        
                        # Load chunk with proper offset and extent
                        comp_data.load_chunk(chunk, offset=[z, y, x], extent=extent)
                        self.series.flush()
                        
                        # Copy to output array
                        data[0, z:z_end, y:y_end, x:x_end] = chunk
                        # print(f"[DEBUG] Loaded chunk at offset {[z, y, x]}, extent {extent}")
                        del chunk
                        gc.collect()

        # Normalize data
        if self.normalize:
            print(f"[DEBUG] Normalizing data for mesh {mesh_name}")
            # Normalize each channel separately to avoid memory issues
            for c in range(data.shape[0]):
                channel_data = data[c]
                data_min = np.min(channel_data)
                data_max = np.max(channel_data)
                if data_max > data_min:
                    channel_data[:] = (channel_data - data_min) / (data_max - data_min)
                else:
                    channel_data[:] = 0.0
            gc.collect()
            print(f"[DEBUG] Normalized data, shape: {data.shape}, size: {data.nbytes / 1e6:.2f} MB")

        return data

    def __iter__(self):
        """Initialize iterator."""
        print("[DEBUG] Initializing iterator")
        self._batch_index = 0
        return self

    def __next__(self):
        """
        Generate a batch of data.
        
        Returns:
            np.ndarray: Batch of shape (batch_size, C, D, H, W) where C=3 for vector fields, C=1 for scalars.
        
        Raises:
            StopIteration: When iteration is complete.
        """
        print("[DEBUG] Generating batch")
        if self._batch_index >= 1:  # Single batch for demo
            print("[DEBUG] Stopping iteration")
            raise StopIteration

        batch = []
        for i in range(self.batch_size):
            print(f"[DEBUG] Processing sample {i+1}/{self.batch_size}")
            sample_data = []
            total_channels = 0
            
            # First pass: calculate total channels needed
            for mesh_name in self.mesh_names:
                if mesh_name in ["B", "E", "J"]:
                    total_channels += 3
                else:
                    total_channels += 1
            
            # Create output sample array
            full_shape = (512, 1792, 512)
            sample = np.empty((total_channels, *full_shape), dtype=np.float32, order='C')
            
            # Second pass: load data directly into sample array
            current_channel = 0
            for mesh_name in self.mesh_names:
                data = self.load_mesh_data(mesh_name, self.iteration)
                num_channels = data.shape[0]
                sample[current_channel:current_channel + num_channels] = data
                current_channel += num_channels
                print(f"[DEBUG] Added mesh {mesh_name}, total channels so far: {current_channel}")
                del data
                gc.collect()
                
            print(f"[DEBUG] Completed sample, shape: {sample.shape}, size: {sample.nbytes / 1e6:.2f} MB")
            batch.append(sample)
            del sample
            gc.collect()

        batch = np.stack(batch, axis=0)
        print(f"[DEBUG] Stacked batch, shape: {batch.shape}, size: {batch.nbytes / 1e6:.2f} MB")
        self._batch_index += 1
        return batch

    def show_meshes(self):
        """Display available meshes and their components."""
        curr_iteration = self.get_iteration(self.iteration)
        print("Available meshes:", list(curr_iteration.meshes))

        for mesh_name in curr_iteration.meshes:
            mesh = curr_iteration.meshes[mesh_name]
            print(f"\n=== Mesh: {mesh_name} ===")
            
            for comp in mesh:
                try:
                    comp_data = mesh[comp]
                    # Use a smaller sample for show_meshes to avoid memory issues
                    sample_shape = (min(64, comp_data.shape[0]), 
                                  min(64, comp_data.shape[1]), 
                                  min(64, comp_data.shape[2]))
                    arr = np.empty(sample_shape, dtype=np.float32, order='C')
                    print(f"[DEBUG] Allocated sample array for {mesh_name}_{comp}, shape: {arr.shape}, size: {arr.nbytes / 1e6:.2f} MB")
                    comp_data.load_chunk(arr, offset=[0, 0, 0], extent=list(sample_shape))
                    self.series.flush()
                    
                    print(f"\n--- Component: {comp} ---")
                    print(f"Full Shape: {comp_data.shape}")
                    print(f"Sample Shape: {arr.shape}")
                    print(f"Dtype     : {arr.dtype}")
                    print(f"First few elements (flat): {arr.ravel()[:10]}")
                    del arr
                except Exception as e:
                    print(f"Error loading {mesh_name}_{comp}: {e}")

    def __del__(self):
        """Clean up series and temporary files."""
        try:
            if self.series is not None:
                self.series.close()
            # Clean up temporary files
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    try:
                        os.remove(os.path.join(self.temp_dir, file))
                    except:
                        pass
                try:
                    os.rmdir(self.temp_dir)
                except:
                    pass
        except Exception as e:
            print(f"Warning: Could not clean up properly: {e}")


if __name__ == "__main__":
    # Example usage with smaller batch size for testing
    # dl = DataLoader(mesh_names=['B', 'E'],  # Start with fewer meshes for testing
                    # batch_size=2, normalize=True, chunk_size=(64, 224, 64))  # Smaller chunks for testing
    dl = DataLoader(mesh_names=['B', 'E', 'J', 'e_all_chargeDensity', 'en_all_chargeDensity', 'n_all_chargeDensity'],  # Start with fewer meshes for testing
                    batch_size=6, normalize=True, chunk_size=(256, 256, 256))  # Smaller chunks for testing
    dl.show_meshes()
    
    # Get a batch
    for batch in dl:
        print(f"Batch shape: {batch.shape}")
        print(f"Batch memory usage: {batch.nbytes / 1e6:.2f} MB")
        break