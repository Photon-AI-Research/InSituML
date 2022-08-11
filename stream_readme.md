## Stream Initialization

On the cluster:
```
/scratch/ws/1/s5960712-ml_streaming/runPIConGPU.sh 'your_stream_path'
```

Check if stream job is submitted:
```
squeue -u $USER
```

Once the job is allocated resources, check if a .sst file is created at
```
'your_stream_path'/simOutput/openPMD/stream.sst
```

## Configure stream reader

Configure stream_config.json in StreamDataReader folder, configure requirements
for eg.
```
{
   "stream_path":"'your_stream_path'/simOutput/openPMD/stream.sst",
   "meshes":[
      "E",
      "J"
   ],
   "particles":{
      "e":[
         "position",
          "id",
          "momentum",
          "weighting"
      ]
   }
}
```

In your code, to access this stream:
```python

streamBuffer = StreamBuffer(buffer_size = buffer_size)

#data is returned as dictionary configured in config file

data_dict = streamBuffer.read_buffer()

#check 'ModelTrainerTaskWise.py' file for example and to parallely fill the buffer after reading it
```

