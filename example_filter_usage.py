"""
Simple example of using the anomaly detector

This is a minimal example showing how to use the filter in your own code.
"""
from service.parser import LogParser
from service.anomaly_detector import HybridFilter

def main():
    print("Step 1: Parsing log file...")
    parser = LogParser()
    events = list(parser.parse_logs('data/sample_log.json'))
    print(f"  ✓ Parsed {len(events)} events")
    
    print("\nStep 2: Creating anomaly detector...")
    detector = HybridFilter(z_score_threshold=1.5)
    print("  ✓ Detector created")
    
    print("\nStep 3: Detecting anomalies...")
    filtered_events, anomalies = detector.filter_and_detect(events)
    print(f"  ✓ Found {len(anomalies)} anomalies")
    
    print("\nStep 4: Results")
    stats = detector.get_combined_stats()
    token_stats = stats['token_savings']
    
    print(f"\n📊 Summary:")
    print(f"  Original events: {len(events)}")
    print(f"  Anomalies detected: {len(anomalies)}")
    print(f"  Token reduction: {token_stats['token_reduction_rate']*100:.1f}%")
    
    print(f"\n🔍 Sample Anomalies:")
    for i, (event, reasons) in enumerate(anomalies[:3], 1):
        print(f"  {i}. {event.event} (Event #{event.event_id})")
    
    return anomalies

if __name__ == '__main__':
    anomalies = main()

