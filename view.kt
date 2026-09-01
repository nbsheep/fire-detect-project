package dji.sampleV5.aircraft.pages;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.fragment.app.activityViewModels;
import dji.sampleV5.aircraft.databinding.FragCodeFlightControlPageBinding;
import dji.sampleV5.aircraft.models.CodeFlightControlVM;
import dji.sampleV5.aircraft.models.SimulatorVM;
import dji.sdk.keyvalue.value.common.EmptyMsg;
import dji.sdk.keyvalue.value.common.LocationCoordinate2D;
import dji.v5.common.callback.CommonCallbacks;
import dji.v5.common.error.IDJIError;
import dji.v5.manager.aircraft.simulator.InitializationSettings;
import dji.v5.ux.core.util.ToastUtils;

/**
 * ============================================================================
 * 代码控制飞行页面（Fragment）
 * ============================================================================
 *
 * 【功能说明】
 * 本页面提供了一系列按钮，通过代码（而非遥控器摇杆）控制无人机执行各种飞行动作，
 * 是无人机自主飞行功能开发和测试的核心交互界面。
 *
 * 【核心能力】
 * 1. 飞行模拟器控制 —— 在无桨无电机的安全环境下验证飞行逻辑
 * 2. 虚拟摇杆控制 —— 通过代码发送速度指令，替代物理摇杆
 * 3. 方向移动控制 —— 前进/后退/左移/右移/上升/下降
 * 4. 一键动作控制 —— 起飞 / 自动降落
 * 5. 速度档位调节 —— 调整移动速度的快慢
 *
 * 【安全操作流程（必须严格遵守）】
 * ┌─────────────────────────────────────────────────────────────────┐
 * │ 第1步：点击「打开飞行模拟器」                                   │
 * │        → 在模拟环境中测试所有方向指令，确认无人机响应正确       │
 * │        → 此时无人机无需安装桨叶，不会造成物理伤害               │
 * ├─────────────────────────────────────────────────────────────────┤
 * │ 第2步：点击「开启虚拟摇杆」                                    │
 * │        → 使能代码控制模式，让应用程序可以发送飞行指令           │
 * ├─────────────────────────────────────────────────────────────────┤
 * │ 第3步：分别点击 前进/后退/上升/下降 等按钮                     │
 * │        → 观察模拟器中的无人机运动方向是否正确                  │
 * │        → 如果方向与预期相反，需要到 ViewModel 中取反修正       │
 * ├─────────────────────────────────────────────────────────────────┤
 * │ 第4步：确认无误后，关闭模拟器，到空旷真机环境进行实飞测试      │
 * └─────────────────────────────────────────────────────────────────┘
 *
 * 【架构说明】
 * CodeFlightControlFragment（本类）  →  UI层：负责按钮点击和界面显示
 *                  ↓ 通过 activityViewModels() 共享数据
 * CodeFlightControlVM（ViewModel）   →  业务层：封装飞行控制逻辑
 *                  ↓
 * DJI SDK（VirtualStick / FlightController）  →  SDK层：与无人机硬件通信
 *
 * @see CodeFlightControlVM 对应的业务逻辑ViewModel
 * @see SimulatorVM 飞行模拟器控制ViewModel
 * ============================================================================
 */
public class CodeFlightControlFragment extends DJIFragment {

    // -------------------- 依赖注入（通过AndroidX ViewModel机制） --------------------

    /**
     * 代码飞行控制ViewModel —— 管理飞行指令、速度档位、状态文字等
     * 使用 activityViewModels() 确保与Activity共享同一实例
     */
    private final CodeFlightControlVM codeFlightControlVM by activityViewModels();

    /**
     * 飞行模拟器ViewModel —— 管理模拟器的开启/关闭
     */
    private final SimulatorVM simulatorVM by activityViewModels();

    /**
     * ViewBinding实例 —— 用于访问布局文件中的所有UI控件
     * 采用懒加载方式，在onCreateView中初始化
     */
    private FragCodeFlightControlPageBinding? binding = null;

    // -------------------- Fragment生命周期回调 --------------------

    /**
     * 创建Fragment的视图层次结构
     * 使用ViewBinding将布局文件 FragCodeFlightControlPage 绑定到当前Fragment
     */
    @Override
    public View onCreateView(
            LayoutInflater inflater,
            ViewGroup container,
            Bundle savedInstanceState
    ) {
        binding = FragCodeFlightControlPageBinding.inflate(inflater, container, false);
        return binding?.root;
    }

    /**
     * 视图创建完成后的初始化工作
     * 1. 为所有按钮设置点击监听器
     * 2. 观察ViewModel中的状态数据，实时更新UI
     */
    @Override
    public void onViewCreated(View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        initListeners();   // 初始化所有按钮的点击事件
        observeState();    // 建立数据观察绑定
    }

    // -------------------- UI状态观察（响应式更新） --------------------

    /**
     * 观察ViewModel中的状态数据，当数据变化时自动更新UI
     *
     * 观察的数据包括：
     * - statusText：状态提示文字（如"就绪"、"飞行中"等）
     * - speedLevel：当前速度档位（1-10 或类似范围）
     */
    private void observeState() {
        // 观察状态文字变化 → 更新到状态TextView
        codeFlightControlVM.statusText.observe(viewLifecycleOwner) {
            binding?.flightStatusTv?.text = it;
        }

        // 观察速度档位变化 → 更新到速度信息TextView
        codeFlightControlVM.speedLevel.observe(viewLifecycleOwner) {
            binding?.speedInfoTv?.text = "速度档: $it";
        }
    }

    // -------------------- 按钮监听器初始化 --------------------

    /**
     * 为所有UI按钮绑定点击事件
     * 每个按钮点击后都会调用ViewModel中对应的业务方法
     *
     * 按钮分类：
     * 1. 模拟器控制（2个按钮）
     * 2. 虚拟摇杆控制（2个按钮）
     * 3. 速度档位控制（2个按钮）
     * 4. 方向移动控制（6个按钮 + 1个停止按钮）
     * 5. 一键动作控制（2个按钮）
     */
    private void initListeners() {

        // ======================== 一、飞行模拟器控制 ========================

        /**
         * 【打开飞行模拟器】
         * 作用：在地面站软件中模拟GPS定位和飞行姿态，无需实际起飞即可测试飞行代码
         * 安全价值：无桨无电机，完全软件模拟，零物理风险
         *
         * 模拟参数：
         * - 位置：深圳（22.5333°N, 113.9308°E）
         * - GPS星数：10颗（模拟良好信号）
         */
        binding?.btnEnableSimulator?.setOnClickListener {
            enableSimulator();
        }

        /**
         * 【关闭飞行模拟器】
         * 作用：退出模拟模式，恢复真实飞行控制
         * 关闭后无人机将读取真实的GPS信号和传感器数据
         */
        binding?.btnDisableSimulator?.setOnClickListener {
            simulatorVM.disableSimulator(object : CommonCallbacks.CompletionCallback {
                @Override
                public void onSuccess() {
                    ToastUtils.showToast("模拟器已关闭");
                }

                @Override
                public void onFailure(IDJIError error) {
                    ToastUtils.showToast("关闭模拟器失败: " + error.description());
                }
            });
        }

        // ======================== 二、虚拟摇杆控制 ========================

        /**
         * 【开启虚拟摇杆】
         * 作用：启用"代码控制"模式，允许应用程序通过SDK发送飞行指令
         * 前置条件：必须先开启此功能，后续移动指令才有效
         *
         * 技术原理：将应用程序作为虚拟的遥控器，通过MAVLink/VirtualStick协议
         * 向飞控系统发送速度/姿态指令
         */
        binding?.btnEnableVirtualStick?.setOnClickListener {
            codeFlightControlVM.enableVirtualStick(object : CommonCallbacks.CompletionCallback {
                @Override
                public void onSuccess() {
                    ToastUtils.showToast("虚拟摇杆已启用");
                }

                @Override
                public void onFailure(IDJIError error) {
                    ToastUtils.showToast("启用失败: " + error.description());
                }
            });
        }

        /**
         * 【禁用虚拟摇杆】
         * 作用：关闭代码控制模式，收回控制权
         * 禁用后，应用程序将无法再发送飞行指令（保证安全，防止意外误触）
         */
        binding?.btnDisableVirtualStick?.setOnClickListener {
            codeFlightControlVM.disableVirtualStick(object : CommonCallbacks.CompletionCallback {
                @Override
                public void onSuccess() {
                    ToastUtils.showToast("虚拟摇杆已禁用");
                }

                @Override
                public void onFailure(IDJIError error) {
                    ToastUtils.showToast("禁用失败: " + error.description());
                }
            });
        }

        // ======================== 三、速度档位控制 ========================

        /**
         * 【加速】
         * 提高移动速度档位（增加速度系数），使无人机移动更快
         * 用于调节不同场景下的飞行速度（如巡检时低速、转场时高速）
         */
        binding?.btnSpeedUp?.setOnClickListener {
            codeFlightControlVM.speedUp();
        }

        /**
         * 【减速】
         * 降低移动速度档位（减少速度系数），使无人机移动更慢
         * 用于精细操控或悬停观察时的低速移动
         */
        binding?.btnSpeedDown?.setOnClickListener {
            codeFlightControlVM.speedDown();
        }

        // ======================== 四、方向移动控制 ========================

        /**
         * 【前移】发送前进指令（机头方向水平向前）
         */
        binding?.btnMoveForward?.setOnClickListener {
            codeFlightControlVM.moveForward();
        }

        /**
         * 【后移】发送后退指令（机头方向水平向后）
         */
        binding?.btnMoveBackward?.setOnClickListener {
            codeFlightControlVM.moveBackward();
        }

        /**
         * 【左移】发送左平移指令（水平向左，不改变机头朝向）
         * 区别于偏航旋转，这是真正的侧向位移
         */
        binding?.btnMoveLeft?.setOnClickListener {
            codeFlightControlVM.moveLeft();
        }

        /**
         * 【右移】发送右平移指令（水平向右，不改变机头朝向）
         */
        binding?.btnMoveRight?.setOnClickListener {
            codeFlightControlVM.moveRight();
        }

        /**
         * 【上升】发送垂直上升指令（增加高度）
         */
        binding?.btnMoveUp?.setOnClickListener {
            codeFlightControlVM.moveUp();
        }

        /**
         * 【下降】发送垂直下降指令（降低高度）
         */
        binding?.btnMoveDown?.setOnClickListener {
            codeFlightControlVM.moveDown();
        }

        /**
         * 【停止移动】发送悬停指令
         * 立即停止所有方向的速度输出，无人机将悬停于当前位置
         * 相当于急停按钮，用于紧急情况或快速定点
         */
        binding?.btnStopMove?.setOnClickListener {
            codeFlightControlVM.stop();
            ToastUtils.showToast("已停止（悬停）");
        }

        // ======================== 五、一键动作控制 ========================

        /**
         * 【起飞】
         * 发送起飞指令，无人机将从地面垂直上升至预设高度（通常1.2-1.5米）
         *
         * 前置条件：
         * - GPS信号良好（≥10颗星）
         * - 无人机处于地面待命状态
         * - 已开启虚拟摇杆（或处于模拟器模式）
         *
         * 回调参数 EmptyMsg 表示成功时无需额外数据
         */
        binding?.btnTakeOff?.setOnClickListener {
            codeFlightControlVM.takeOff(object :
                    CommonCallbacks.CompletionCallbackWithParam<EmptyMsg> {
                @Override
                public void onSuccess(EmptyMsg t) {
                    ToastUtils.showToast("起飞指令已下发");
                }

                @Override
                public void onFailure(IDJIError error) {
                    ToastUtils.showToast("起飞失败: " + error.description());
                }
            });
        }

        /**
         * 【自动降落】
         * 发送降落指令，无人机将自动执行降落程序
         * 降落过程包括：悬停→缓降→触地检测→电机停转
         *
         * 安全注意：
         * - 降落前确保降落区域平坦无障碍物
         * - 如遇紧急情况可取消降落（需额外实现）
         */
        binding?.btnLanding?.setOnClickListener {
            codeFlightControlVM.autoLanding(object :
                    CommonCallbacks.CompletionCallbackWithParam<EmptyMsg> {
                @Override
                public void onSuccess(EmptyMsg t) {
                    ToastUtils.showToast("降落指令已下发");
                }

                @Override
                public void onFailure(IDJIError error) {
                    ToastUtils.showToast("降落失败: " + error.description());
                }
            });
        }
    }

    // -------------------- 私有辅助方法 --------------------

    /**
     * 打开飞行模拟器
     * 模拟器会在软件层面模拟GPS定位、惯导数据、飞行姿态等
     * 让开发者可以在无桨无电机的安全环境下测试飞行逻辑
     *
     * 初始化参数：
     *   - 坐标：深圳（北纬22.5333°，东经113.9308°）
     *   - GPS星数：10颗（保证定位精度）
     *
     * 使用场景：
     * 1. 测试飞行代码逻辑是否正确
     * 2. 验证方向指令是否与预期一致
     * 3. 学生教学演示（零风险）
     */
    private void enableSimulator() {
        // 构造模拟位置：深圳坐标
        LocationCoordinate2D coordinate = new LocationCoordinate2D(22.5333, 113.9308);
        // 创建模拟器初始化配置（位置 + GPS星数）
        InitializationSettings settings = InitializationSettings.createInstance(coordinate, 10);

        // 调用ViewModel启用模拟器
        simulatorVM.enableSimulator(settings, new CommonCallbacks.CompletionCallback() {
            @Override
            public void onSuccess() {
                ToastUtils.showToast("模拟器已打开，可以安全测试飞行代码");
            }

            @Override
            public void onFailure(IDJIError error) {
                ToastUtils.showToast("打开模拟器失败: " + error.description());
            }
        });
    }
}