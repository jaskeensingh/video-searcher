from video_searcher import VideoAnalyzer
import json

def process_batch(video_paths):
    """Example of batch processing multiple videos"""
    analyzer = VideoAnalyzer()
    
    results = []
    for path in video_paths:
        try:
            metadata = analyzer.process_video(path)
            results.append({
                'path': path,
                'status': 'success',
                'metadata': metadata
            })
        except Exception as e:
            results.append({
                'path': path,
                'status': 'error',
                'error': str(e)
            })
    
    # Save results
    with open('batch_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    videos = [
        'examples/data/video1.mp4',
        'examples/data/video2.mp4'
    ]
    process_batch(videos)