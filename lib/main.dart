import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:camera/camera.dart';
import 'package:pytorch_lite/pytorch_lite.dart'; // 確保有 import
import 'package:onnxruntime/onnxruntime.dart';
import 'dart:typed_data'; // 解決 Float32List 未定義
import 'package:image/image.dart' as img; // 解決 img 和 Image 未定義
import 'package:flutter/semantics.dart';
import 'dart:ui' as ui;
import 'package:seg_vision_app/debug_dashboard.dart'; // 檔名要對應

// If you plan to preview camera & map, add these packages to pubspec.yaml and wire them up:
// camera: ^0.11.0
// google_maps_flutter: ^2.7.0
// geolocator: ^13.0.1
// For this UI skeleton, camera preview shows a placeholder until you integrate the real controller.
// Map page is included but centers on a fallback LatLng until you connect location services.

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const BlindVisionApp());
}

class BlindVisionApp extends StatelessWidget {
  const BlindVisionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Blind Vision Assistant',
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: Colors.blue,
        brightness: Brightness.dark,
        textTheme: const TextTheme(
          bodyLarge: TextStyle(fontSize: 18),
          bodyMedium: TextStyle(fontSize: 16),
          titleLarge: TextStyle(fontSize: 22, fontWeight: FontWeight.w700),
        ),
        // High-contrast, large tap targets
        visualDensity: VisualDensity.comfortable,
        splashFactory: InkRipple.splashFactory,
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final _tts = FlutterTts();
  bool _isDetecting = false;
  bool _muteVoice = false;
  Timer? _timer;
  DetectionResult? _last;
  DateTime? _lastFrameTime;
  CameraController? _cameraController; // 新增這行
  List<CameraDescription>? _cameras;    // 新增這行
  bool _isCameraInitialized = false;    // 新增這行
  OrtSession? _onnxSession;
  bool _isModelLoaded = false;

  @override
  void initState() {
    super.initState();
    _initTts();
    _initCamera();
  }
  Future<void> _initCamera() async {
    // 1. 請求權限
    await [Permission.camera, Permission.locationWhenInUse].request();

    // 2. 取得可用相機
    _cameras = await availableCameras();
    if (_cameras == null || _cameras!.isEmpty) return;

    // 3. 設定相機控制器 (使用後置鏡頭)
    _cameraController = CameraController(
      _cameras![0],
      ResolutionPreset.medium,
      enableAudio: false,
    );

    try {
      await _cameraController!.initialize();
      if (!mounted) return;
      await _loadModel();
      setState(() {
        _isCameraInitialized = true;
      });
    } catch (e) {
      debugPrint("相機初始化失敗: $e");
    }
  }

  Future<void> _loadModel() async {
    try {
      // 1. 初始化 ONNX 環境
      OrtEnv.instance.init();

      // 2. 從 Assets 讀取模型 (路徑要跟你的 pubspec.yaml 一致)
      final modelBytes = await rootBundle.load('assets/models/segformer_for_mobile.onnx');

      // 3. 建立 Session
      final sessionOptions = OrtSessionOptions();
      _onnxSession = OrtSession.fromBuffer(modelBytes.buffer.asUint8List(), sessionOptions);

      setState(() => _isModelLoaded = true);
      debugPrint("ONNX 模型載入成功");
    } catch (e) {
      debugPrint("ONNX 模型載入失敗: $e");
    }
  }


  Future<void> _initTts() async {
    await _tts.setLanguage('zh-TW');
    await _tts.setSpeechRate(0.45);
    await _tts.setVolume(1.0);
    await _tts.setPitch(1.0);
  }

  Future<void> _requestPermissions() async {
    await [Permission.camera, Permission.locationWhenInUse].request();
    // Optionally: microphone, if you want voice commands.
  }

  @override
  void dispose() {
    _timer?.cancel();
    _tts.stop();
    _cameraController?.dispose();
    _onnxSession?.release(); // 新增這行
    super.dispose();
  }

  void _processRealDetections(dynamic results) {
    if (results == null || results is! List) return;

    List<DetectionObject> objects = [];

    for (var res in results) {
      if (res != null) {
        // 使用 dynamic 存取，避開型別定義錯誤
        final String label = res.className?.toString() ?? 'unknown';
        final double score = double.tryParse(res.score?.toString() ?? '0.0') ?? 0.0;

        objects.add(DetectionObject(
          label: label,
          direction: 'ahead',
          confidence: score,
        ));
      }
    }

    final newResult = DetectionResult(
      timestamp: DateTime.now(),
      objects: objects,
    );

    _handleDetections(newResult);

    setState(() {
      _last = newResult;
      _lastFrameTime = DateTime.now();
    });
  }

  void _startDetection() {
    // 1. 如果已經在偵測、相機沒好或模型沒載入，就直接返回
    if (_isDetecting || _cameraController == null || !_isModelLoaded) {
      debugPrint("無法開始偵測：相機或模型尚未就緒");
      return;
    }

    // 2. 關鍵：更新 UI 狀態為「偵測中」
    setState(() {
      _isDetecting = true;
    });

    debugPrint("開始偵測循環...");

    _timer = Timer.periodic(const Duration(seconds: 5), (timer) async {
      // 這裡的判斷會依賴上面 setState 的結果
      if (!_isDetecting || _onnxSession == null) return;

      try {
        debugPrint("正在擷取畫面進行辨識...");
        XFile file = await _cameraController!.takePicture();
        Uint8List bytes = await file.readAsBytes();

        // ... 影像前處理部分保持不變 ...
        img.Image? baseImage = img.decodeImage(bytes);
        if (baseImage == null) return;
        img.Image resized = img.copyResize(baseImage, width: 512, height: 512);

        var floatBuffer = Float32List(1 * 3 * 512 * 512);
        for (int y = 0; y < 512; y++) {
          for (int x = 0; x < 512; x++) {
            var pixel = resized.getPixel(x, y);
            // 關鍵：這裡必須跟 Debug 模式完全一樣，加入 ImageNet 歸一化
            floatBuffer[0 * 512 * 512 + y * 512 + x] = (pixel.r / 255.0 - 0.485) / 0.229;
            floatBuffer[1 * 512 * 512 + y * 512 + x] = (pixel.g / 255.0 - 0.456) / 0.224;
            floatBuffer[2 * 512 * 512 + y * 512 + x] = (pixel.b / 255.0 - 0.406) / 0.225;

          }
        }

        final inputName = _onnxSession!.inputNames.first;
        // 確保這裡也是使用同樣的建立方法
        final inputTensor = OrtValueTensor.createTensorWithDataList(floatBuffer, [1, 3, 512, 512]);
        final outputs = await _onnxSession!.run(OrtRunOptions(), {inputName: inputTensor});

        if (outputs != null && outputs.isNotEmpty) {
          final dynamic data = outputs[0]?.value;
          if (data is List && data.isNotEmpty) {
            int zebraCount = 0;

            // 1. 取得第一張圖的 Mask
            final List<dynamic> mask = data[0];

            // 2. 全掃描（確保不論斑馬線在哪裡都抓得到）
            for (var row in mask) {
              if (row is List) {
                for (var pixel in row) {
                  // 使用 toInt() 確保與 1 比較時型別正確
                  if (pixel.toInt() == 1) {
                    zebraCount++;
                  }
                }
              }
            }

            debugPrint("主畫面偵測結果 - 斑馬線總像素點: $zebraCount");

            String resultLabel = "safe";
            // 3. 設定一個合理的門檻
            // 512x512 = 262,144 像素。如果斑馬線佔畫面 1%，點數約 2600 點
            // 先設 500 點作為測試門檻
            if (zebraCount > 500) {
              resultLabel = "zebra_crossing";
            }

            _updateStatus(resultLabel);
          }
        }
        inputTensor.release();
      } catch (e) {
        debugPrint("主畫面辨識出錯: $e");
      }
    });
  }
  void _updateStatus(String label) {
    String message = "";
    if (label == "zebra_crossing") {
      message = "注意，前方有斑馬線。";
    } else if (label == "car") {
      message = "注意，前方有車輛。";
    }

    setState(() {
      _lastFrameTime = DateTime.now();
      _last = DetectionResult(
          timestamp: DateTime.now(),
          objects: [DetectionObject(label: label, direction: 'ahead', confidence: 1.0)]
      );
    });

    if (message.isNotEmpty && !_muteVoice) {
      _tts.speak(message);
    }
  }

  void _stopDetection() {
    _timer?.cancel();
    _timer = null;
    setState(() => _isDetecting = false);
  }

  Future<void> _handleDetections(DetectionResult result) async {
    setState(() => _last = result);

    final top = result.topPriorityMessage();

    // Announce via system accessibility (TalkBack/VoiceOver) + TTS + haptics.
    if (top != null) {
      // System announce (respects screen reader)
      // NOTE: Works best when a11y services are enabled on device.
      SemanticsService.announce(top, TextDirection.ltr);

      if (!_muteVoice) {
        await _tts.stop();
        await _tts.speak(top);
      }

      //if (await Vibration.hasVibrator() ?? false) {
      //  Vibration.vibrate(pattern: [0, 250, 150, 300]);
      //}
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Blind Vision Assistant'),
        actions: [
          // 只需要留一個除錯按鈕即可，記得按鈕之間要有逗號
          IconButton(
            icon: const Icon(Icons.bug_report, color: Colors.orange),
            onPressed: () {
              // 這裡加入判斷，確保相機跟模型都準備好了才跳轉
              if (_cameraController != null && _cameraController!.value.isInitialized && _onnxSession != null) {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => DebugDashboard(
                      controller: _cameraController!,
                      session: _onnxSession!,
                    ),
                  ),
                );
              } else {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text("相機或模型尚未就緒，請稍候")),
                );
              }
            },
          ), // <--- 這個逗號非常重要，用來分隔 actions 列表中的元素
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            // Top status & last message banner
            _StatusBar(
              isDetecting: _isDetecting,
              last: _last,
              lastFrameTime: _lastFrameTime,
            ),
            // Camera + overlay
            Expanded(
              child: Stack(
                children: [
                  _isCameraInitialized
                      ? CameraPreview(_cameraController!)
                      : _CameraPreviewPlaceholder(isReady: _isDetecting),
                  _DetectionOverlay(result: _last),
                ],
              ),
            ),
            // Controls
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
              child: Row(
                children: [
                  Expanded(
                    child: Semantics(
                      button: true,
                      label: _isDetecting ? '停止偵測' : '開始偵測',
                      child: ElevatedButton.icon(
                        onPressed: _isDetecting ? _stopDetection : _startDetection,
                        icon: Icon(_isDetecting ? Icons.stop : Icons.play_arrow),
                        style: ElevatedButton.styleFrom(minimumSize: const Size.fromHeight(56)),
                        label: Text(_isDetecting ? '停止偵測' : '開始偵測'),
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Semantics(
                      button: true,
                      label: '開啟地圖',
                      child: OutlinedButton.icon(
                        onPressed: () async {
                          await Navigator.of(context).push(
                            MaterialPageRoute(builder: (_) => const MapPage()),
                          );
                        },
                        icon: const Icon(Icons.map),
                        style: OutlinedButton.styleFrom(minimumSize: const Size.fromHeight(56)),
                        label: const Text('地圖'),
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Semantics(
                    button: true,
                    label: _muteVoice ? '開啟語音播報' : '靜音',
                    child: IconButton(
                      onPressed: () => setState(() => _muteVoice = !_muteVoice),
                      iconSize: 32,
                      tooltip: _muteVoice ? '開啟語音播報' : '靜音',
                      icon: Icon(_muteVoice ? Icons.volume_off : Icons.volume_up),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _StatusBar extends StatelessWidget {
  const _StatusBar({
    required this.isDetecting,
    required this.last,
    required this.lastFrameTime,
  });

  final bool isDetecting;
  final DetectionResult? last;
  final DateTime? lastFrameTime;

  @override
  Widget build(BuildContext context) {
    final text = isDetecting ? '偵測中…' : '待機';
    final color = isDetecting ? Colors.green : Colors.grey;

    final lastMsg = last?.topPriorityMessage() ?? '尚無偵測結果';
    final timeStr = lastFrameTime == null
        ? ''
        : '  ·  上次更新 ${TimeOfDay.fromDateTime(lastFrameTime!).format(context)}';

    return Semantics(
      container: true,
      label: '目前狀態 $text。$lastMsg',
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        color: color.withOpacity(0.15),
        child: Row(
          children: [
            Container(width: 10, height: 10, decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                '$text｜$lastMsg$timeStr',
                style: Theme.of(context)
                    .textTheme
                    .titleLarge
                    ?.copyWith(letterSpacing: .5),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _CameraPreviewPlaceholder extends StatelessWidget {
  const _CameraPreviewPlaceholder({required this.isReady});
  final bool isReady;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      label: '相機預覽',
      child: Container(
        alignment: Alignment.center,
        decoration: BoxDecoration(
          gradient: LinearGradient(colors: [
            Colors.black.withOpacity(0.9),
            Colors.blueGrey.shade900,
          ], begin: Alignment.topLeft, end: Alignment.bottomRight),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(isReady ? Icons.videocam : Icons.videocam_off, size: 72),
            const SizedBox(height: 12),
            Text(isReady ? '相機連線中…' : '相機尚未啟動', style: const TextStyle(fontSize: 18)),
            const SizedBox(height: 4),
            const Text('（整合 camera 套件後會顯示即時畫面）'),
          ],
        ),
      ),
    );
  }
}

class _DetectionOverlay extends StatelessWidget {
  const _DetectionOverlay({this.result});
  final DetectionResult? result;

  @override
  Widget build(BuildContext context) {
    final res = result;
    if (res == null || res.objects.isEmpty) return const SizedBox.shrink();

    // A pill-style overlay list in the upper area.
    return IgnorePointer(
      child: Align(
        alignment: Alignment.topCenter,
        child: Padding(
          padding: const EdgeInsets.only(top: 24.0),
          child: Wrap(
            spacing: 8,
            runSpacing: 8,
            children: res.objects.map((o) {
              final icon = o.label == 'zebra_crossing'
                  ? Icons.directions_walk
                  : Icons.directions_car;
              final dirText = _dirText(o.direction);
              return Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.55),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: Colors.white24),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(icon, size: 20),
                    const SizedBox(width: 6),
                    Text('${_labelText(o.label)}｜$dirText'),
                    const SizedBox(width: 6),
                    Text('(${(o.confidence * 100).toStringAsFixed(0)}%)', style: const TextStyle(fontFeatures: [FontFeature.tabularFigures()])),
                  ],
                ),
              );
            }).toList(),
          ),
        ),
      ),
    );
  }

  static String _labelText(String raw) {
    switch (raw) {
      case 'zebra_crossing':
        return '斑馬線';
      case 'car':
        return '車輛';
      default:
        return raw;
    }
  }

  static String _dirText(String raw) {
    switch (raw) {
      case 'left':
        return '左側';
      case 'right':
        return '右側';
      case 'ahead':
        return '前方';
      case 'right_ahead':
        return '右前方';
      case 'left_ahead':
        return '左前方';
      default:
        return raw;
    }
  }
}

class MapPage extends StatelessWidget {
  const MapPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('附近路況（地圖）')),
      body: const _MapPlaceholder(),
    );
  }
}

class _MapPlaceholder extends StatelessWidget {
  const _MapPlaceholder();

  @override
  Widget build(BuildContext context) {
    // Replace with GoogleMap widget once API Key is set.
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: const [
          Icon(Icons.map, size: 72),
          SizedBox(height: 12),
          Text('這裡會顯示 Google Map'),
          SizedBox(height: 4),
          Text('（設定 API Key 後用 google_maps_flutter 取代）'),
        ],
      ),
    );
  }
}

// ======= Data Model & Parser =======
class DetectionObject {
  final String label; // e.g., 'zebra_crossing', 'car'
  final String direction; // e.g., 'right_ahead', 'left', 'ahead'
  final double confidence; // 0~1
  const DetectionObject({required this.label, required this.direction, required this.confidence});

  factory DetectionObject.fromJson(Map<String, dynamic> m) => DetectionObject(
    label: (m['label'] ?? '').toString(),
    direction: (m['direction'] ?? '').toString(),
    confidence: (m['confidence'] ?? 0).toDouble(),
  );
}

class DetectionResult {
  final DateTime timestamp;
  final List<DetectionObject> objects;
  const DetectionResult({required this.timestamp, required this.objects});

  /// Convert top priority object to an announcement string in Chinese.
  /// Priority rule (you can tweak): zebra_crossing > car; higher confidence first.
  String? topPriorityMessage() {
    if (objects.isEmpty) return null;
    final sorted = [...objects]
      ..sort((a, b) {
        final priA = a.label == 'zebra_crossing' ? 1 : 0;
        final priB = b.label == 'zebra_crossing' ? 1 : 0;
        final c = priB.compareTo(priA);
        if (c != 0) return c;
        return b.confidence.compareTo(a.confidence);
      });
    final top = sorted.first;
    final label = top.label == 'zebra_crossing' ? '斑馬線' : (top.label == 'car' ? '車輛' : top.label);

    String dir;
    switch (top.direction) {
      case 'right':
        dir = '右側';
        break;
      case 'left':
        dir = '左側';
        break;
      case 'ahead':
        dir = '前方';
        break;
      case 'right_ahead':
        dir = '右前方';
        break;
      case 'left_ahead':
        dir = '左前方';
        break;
      default:
        dir = top.direction;
    }
    return '注意，$dir有$label。';
  }
}

DetectionResult parseJsonFrame(String jsonStr) {
  final m = jsonDecode(jsonStr) as Map<String, dynamic>;
  final ts = DateTime.tryParse((m['timestamp'] ?? '').toString()) ?? DateTime.now();
  final list = (m['objects'] as List<dynamic>? ?? [])
      .map((e) => DetectionObject.fromJson(e as Map<String, dynamic>))
      .toList();
  return DetectionResult(timestamp: ts, objects: list);
}

/* =========================
HOW TO INTEGRATE (quick notes)

pubspec.yaml (add and `flutter pub get`):

dependencies:
  flutter:
    sdk: flutter
  flutter_tts: ^3.8.5
  permission_handler: ^11.3.1
  vibration: ^1.9.0
  # For live camera preview & frame extraction:
  camera: ^0.11.0
  # For map:
  google_maps_flutter: ^2.7.0

Android setup:
- Add your Google Maps API key to android/app/src/main/AndroidManifest.xml:
  <meta-data android:name="com.google.android.geo.API_KEY" android:value="YOUR_API_KEY"/>
- Request camera & location permissions in the Manifest as required by packages.

Camera wiring (replace placeholder):
- Create a CameraController with the back camera.
- Start video stream or periodic image stream; on every 5 seconds, pull one frame (or the latest)
  and send to your detection service; parse returned JSON; call _handleDetections().

Accessibility tips:
- Keep buttons big (56px+) and labels clear.
- Always provide Semantics labels for non-text UI.
- Use TTS + SemanticsService.announce for real-time alerts.
- Provide a mute toggle and concise, consistent phrasing (e.g., 「注意，右前方有斑馬線。」).

========================= */
