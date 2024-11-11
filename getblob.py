import blobconverter

with open('sciezka_do_blobow.txt', 'w') as f:
	f.writelines(f'EAST detection: {blobconverter.from_zoo(name='east_text_detection_256x256', zoo_type='depthai', shaves=6, version='2021.2')}')
	f.write(f'Recognition: {blobconverter.from_zoo(name='text-recognition-0012', shaves=6, version='2021.2')}')

# ten skrypt pobiera pliki .blob. z utowrzonego pliku 'sciezka_do_blobów.txt' trzeba skopiować ścieżki do plików a same pliki skopiować do folderu models