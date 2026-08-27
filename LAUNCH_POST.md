Hey guys. I run a lot of local models and I was getting really tired of keeping two or three terminal windows open just to see what was going on. I always had nvtop running to see my GPU usage, and then I had to constantly stare at the scrolling llama.cpp logs to figure out my tokens per second or check if my KV cache was filling up. 

I looked around for a simple tool that combined everything into one clean terminal dashboard, but the stuff I found was either a massive Node framework that ran in a browser or it only worked for NVIDIA cards.

So I decided to just write my own. I called it llamagputop. 

It is a single Python script with literally zero dependencies. You don't need to pip install anything or run docker containers. You just download the file and run it. It reads the system files directly and auto detects if you have AMD, Intel, or NVIDIA GPUs and monitors them all in the same view.

But the part I like the most of it, is the llama.cpp integration. It automatically detects any llama server running in the background and shows you the live prefill speed, generation speed with rolling medians, the exact KV cache percentage, and even which draft head is being used for speculative decoding. It also reads the file descriptors to show exactly how much VRAM each model process is using so you can see if you are spilling over into your system RAM.

I just pushed it to Github under the MIT license. Feel free to grab the code and try it out. The repo is at https://github.com/XscannedX/llamagputop

I would love to hear what you guys think, especially if you run different hardware setups than me. Happy inferencing!
