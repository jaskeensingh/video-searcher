from video_searcher import VideoAnalyzer

def main():
    # Initialize analyzer
    analyzer = VideoAnalyzer()
    
    # Process video
    video_path = 'examples/data/sample.mp4'
    metadata = analyzer.process_video(video_path)
    
    # Search for content
    results = analyzer.search('person walking')
    
    # Print results
    for result in results:
        print(f"Match in {result['video_path']}")
        print(f"Confidence: {result['relevance_score']:.2f}")
        print(f"Type: {result['match_type']}")
        print("---")

if __name__ == '__main__':
    main()