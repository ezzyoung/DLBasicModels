import os
from PIL import Image
from collections import defaultdict
import statistics

def check_image_sizes(folder_path):
    """princess 폴더 내 모든 이미지의 크기를 확인합니다."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    size_stats = defaultdict(list)
    all_sizes = []
    errors = []
    
    # 모든 하위 폴더 탐색
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            if file_ext in image_extensions:
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        size_tuple = (width, height)
                        size_stats[size_tuple].append(file_path)
                        all_sizes.append((width, height, file_path))
                except Exception as e:
                    errors.append((file_path, str(e)))
    
    # 결과 출력
    print("=" * 80)
    print(f"이미지 크기 분석 결과: {folder_path}")
    print("=" * 80)
    print(f"\n총 이미지 개수: {len(all_sizes)}")
    
    if all_sizes:
        widths = [size[0] for size in all_sizes]
        heights = [size[1] for size in all_sizes]
        
        print(f"\n너비(Width) 통계:")
        print(f"  최소: {min(widths)}px")
        print(f"  최대: {max(widths)}px")
        print(f"  평균: {statistics.mean(widths):.2f}px")
        print(f"  중앙값: {statistics.median(widths):.2f}px")
        
        print(f"\n높이(Height) 통계:")
        print(f"  최소: {min(heights)}px")
        print(f"  최대: {max(heights)}px")
        print(f"  평균: {statistics.mean(heights):.2f}px")
        print(f"  중앙값: {statistics.median(heights):.2f}px")
        
        # 가장 많이 사용되는 크기들
        print(f"\n가장 많이 사용되는 이미지 크기 (상위 10개):")
        sorted_sizes = sorted(size_stats.items(), key=lambda x: len(x[1]), reverse=True)
        for i, (size, files) in enumerate(sorted_sizes[:10], 1):
            print(f"  {i}. {size[0]}x{size[1]}px: {len(files)}개 이미지")
        
        # 크기별 분포
        print(f"\n크기별 분포:")
        for size, files in sorted_sizes[:10]:
            print(f"  {size[0]}x{size[1]}: {len(files)}개")
        
        # 각 하위 폴더별 통계
        print(f"\n하위 폴더별 통계:")
        folder_stats = defaultdict(list)
        for width, height, file_path in all_sizes:
            folder_name = os.path.basename(os.path.dirname(file_path))
            folder_stats[folder_name].append((width, height))
        
        for folder_name in sorted(folder_stats.keys()):
            sizes = folder_stats[folder_name]
            widths = [s[0] for s in sizes]
            heights = [s[1] for s in sizes]
            print(f"\n  [{folder_name}]")
            print(f"    이미지 개수: {len(sizes)}")
            print(f"    크기 범위: {min(widths)}x{min(heights)} ~ {max(widths)}x{max(heights)}")
            print(f"    평균 크기: {statistics.mean(widths):.0f}x{statistics.mean(heights):.0f}")
    
    if errors:
        print(f"\n⚠️  읽기 실패한 파일 ({len(errors)}개):")
        for file_path, error in errors[:10]:  # 최대 10개만 표시
            print(f"  {file_path}: {error}")
        if len(errors) > 10:
            print(f"  ... 외 {len(errors) - 10}개 파일")

if __name__ == "__main__":
    princess_folder = os.path.join(os.path.dirname(__file__), "princess")
    check_image_sizes(princess_folder)

