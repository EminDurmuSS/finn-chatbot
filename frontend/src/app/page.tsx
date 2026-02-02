"use client";

import { AutosizeTextarea } from "@/components/ui/autosize-textarea";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  buildDownloadUrl,
  chatMessage,
  confirmAction,
  CopilotActionPlan,
  CopilotActionResult,
  CopilotChatResponse,
  extractFilename,
} from "@/lib/statementCopilot";
import { KeyboardEvent, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type DownloadLink = {
  label: string;
  url: string;
};

interface Message {
  id: string;
  message: string;
  type: "bot" | "user";
  isThinking?: boolean;
  actionPlan?: CopilotActionPlan;
  actionResult?: CopilotActionResult;
  warnings?: string[];
  suggestions?: string[];
  downloads?: DownloadLink[];
}

const DEFAULT_TENANT_ID =
  process.env.NEXT_PUBLIC_STATEMENT_COPILOT_TENANT_ID ?? "";

const createMessageId = () => {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `msg_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
};

const formatBotMessage = (data: CopilotChatResponse) => {
  if (data.error) {
    return data.error;
  }
  const answer = data.answer?.trim();
  if (answer) {
    return answer;
  }
  if (data.action_plan?.human_plan) {
    return data.action_plan.human_plan;
  }
  return "Sorry, I could not generate a response.";
};

const buildDownloads = (actionResult?: CopilotActionResult): DownloadLink[] => {
  if (!actionResult?.artifacts) {
    return [];
  }

  return Object.entries(actionResult.artifacts)
    .map(([key, value]) => {
      if (typeof value !== "string") {
        return null;
      }
      const filename = extractFilename(value);
      const url = buildDownloadUrl(value);
      if (!filename || !url) {
        return null;
      }
      const labelBase = key.replace(/_/g, " ");
      const label = labelBase.includes("file")
        ? filename
        : `${labelBase} (${filename})`;
      return { label, url };
    })
    .filter(Boolean) as DownloadLink[];
};

export default function Chat() {
  const scrollRef = useRef<null | HTMLDivElement>(null);
  const messagesEndRef = useRef<null | HTMLDivElement>(null);
  const [userInput, setUserInput] = useState("");
  const [conversation, setConversation] = useState<Message[]>([]);
  const [userId, setUserId] = useState<string>("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [pendingAction, setPendingAction] = useState<CopilotActionPlan | null>(
    null
  );
  const [isSending, setIsSending] = useState(false);
  const [isConfirming, setIsConfirming] = useState(false);

  const isInputDisabled = isSending || isConfirming || pendingAction !== null;
  const canSend = userInput.trim().length > 0 && !isInputDisabled;
  const inputPlaceholder = pendingAction
    ? "Confirm or reject the action to continue..."
    : "Ask about your expenses...";

  useEffect(() => {
    // Generate a unique user ID when component mounts
    const timestamp = Date.now().toString(36);
    const random = Math.random().toString(36).substring(2, 8);
    const newUserId = `user_${timestamp}_${random}`;
    setUserId(newUserId);

    setConversation([
      {
        id: createMessageId(),
        message: "Hi, I am Finn! 👋 What would you like to know about your expenses?",
        type: "bot",
      },
    ]);
  }, []);

  const maybeAutoScroll = (isUserMessage: boolean) => {
    if (isUserMessage) {
      setTimeout(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
      }, 100);
      return;
    }

    const messageEndPosition =
      messagesEndRef.current?.getBoundingClientRect()?.top || 0;
    const scrollAreaPosition =
      scrollRef.current?.getBoundingClientRect()?.top || 0;
    const scrollAreaHeight = scrollRef.current?.clientHeight || 0;
    const scrollPosition = messageEndPosition - scrollAreaPosition;
    if (scrollAreaHeight - scrollPosition >= -200) {
      setTimeout(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
      }, 100);
    }
  };

  const addMessage = (message: Omit<Message, "id">) => {
    const id = createMessageId();
    setConversation((oldArray) => [...oldArray, { id, ...message }]);
    maybeAutoScroll(message.type === "user");
    return id;
  };

  const replaceMessage = (id: string, message: Omit<Message, "id">) => {
    setConversation((oldArray) =>
      oldArray.map((item) => (item.id === id ? { id, ...message } : item))
    );
    maybeAutoScroll(message.type === "user");
  };

  const ensureUserId = () => {
    if (userId) {
      return userId;
    }
    const timestamp = Date.now().toString(36);
    const random = Math.random().toString(36).substring(2, 8);
    const newUserId = `user_${timestamp}_${random}`;
    setUserId(newUserId);
    return newUserId;
  };

  const handleChatResponse = (data: CopilotChatResponse, messageId: string) => {
    if (data.session_id && data.session_id !== sessionId) {
      setSessionId(data.session_id);
    }

    const downloads = buildDownloads(data.action_result);
    replaceMessage(messageId, {
      message: formatBotMessage(data),
      type: "bot",
      isThinking: false,
      actionPlan: data.action_plan,
      actionResult: data.action_result,
      warnings: data.warnings,
      suggestions: data.suggestions,
      downloads,
    });

    if (data.needs_confirmation && data.action_plan) {
      setPendingAction(data.action_plan);
    } else {
      setPendingAction(null);
    }
  };

  const sendMessage = async (overrideInput?: string) => {
    const messageText = (overrideInput ?? userInput).trim();
    if (!messageText || isInputDisabled) {
      return;
    }

    addMessage({ message: messageText, type: "user" });
    if (!overrideInput) {
      setUserInput("");
    }
    const thinkingId = addMessage({
      message: "...",
      type: "bot",
      isThinking: true,
    });

    setIsSending(true);
    try {
      const data = await chatMessage({
        message: messageText,
        session_id: sessionId,
        tenant_id: DEFAULT_TENANT_ID || null,
        user_id: ensureUserId(),
      });
      handleChatResponse(data, thinkingId);
    } catch (error) {
      const errorMessage =
        error instanceof Error
          ? error.message
          : "Sorry, there was an error processing your message.";
      replaceMessage(thinkingId, {
        message: errorMessage,
        type: "bot",
        isThinking: false,
      });
    } finally {
      setIsSending(false);
    }
  };

  const handleConfirmAction = async (approved: boolean) => {
    if (!pendingAction || !sessionId || isConfirming) {
      return;
    }

    addMessage({
      message: approved ? "Onayliyorum." : "Reddediyorum.",
      type: "user",
    });
    const thinkingId = addMessage({
      message: "...",
      type: "bot",
      isThinking: true,
    });

    setIsConfirming(true);
    try {
      const data = await confirmAction({
        session_id: sessionId,
        action_id: pendingAction.action_id,
        approved,
      });
      setPendingAction(null);
      handleChatResponse(data, thinkingId);
    } catch (error) {
      const errorMessage =
        error instanceof Error
          ? error.message
          : "Sorry, there was an error while confirming the action.";
      replaceMessage(thinkingId, {
        message: errorMessage,
        type: "bot",
        isThinking: false,
      });
      setPendingAction(null);
    } finally {
      setIsConfirming(false);
    }
  };

  const handleEnter = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <main className="h-screen flex flex-row" style={{ background: "#181614" }}>
      {/* Sidebar - hidden on mobile, visible on md and up */}
      <aside className="hidden md:flex w-64 flex-col justify-between p-6 border-r border-white/5 bg-[#181614]">
        <div className="flex justify-start">
          <img src="/bunq-logo.svg" alt="bunq" className="h-8" />
        </div>
        <div className="text-white/20 text-xs font-light">
          Developed by Emin Durmuş
        </div>
      </aside>

      {/* Main Content Wrapper */}
      <div className="flex-1 flex flex-col h-full relative">
        {/* App top bar */}
        <div className="w-full flex justify-center items-center py-4 mb-2">
          <img
            src="/Finn_Circle.png"
            alt="Expense assistant"
            className="w-16 h-16 mr-4"
            style={{ borderRadius: "50%" }}
          />
          <div className="flex flex-col justify-center">
            <span className="text-white text-2xl font-bold leading-tight">
              Finn
            </span>
            <span className="text-white text-sm font-light tracking-wide mt-1">
              Your personal expense assistant
            </span>
          </div>
        </div>
        <ScrollArea ref={scrollRef} className="flex-1 overflow-x-hidden">
          <div className="flex flex-col gap-1 p-2 max-w-3xl mx-auto">
            {conversation.map((msg) => {
              const showConfirmation =
                msg.actionPlan &&
                pendingAction &&
                msg.actionPlan.action_id === pendingAction.action_id;
              const showPlanText =
                msg.actionPlan?.human_plan &&
                msg.actionPlan.human_plan !== msg.message;
              return (
                <div key={msg.id} className="flex gap-2 first:mt-2">
                  {msg.type === "bot" ? (
                    <div
                      className="w-full overflow-hidden p-4 rounded-[20px] text-white relative font-medium max-w-[60%] mr-auto"
                      style={{
                        border: "3px solid transparent",
                        borderRadius: "20px",
                        background:
                          "linear-gradient(#2C2C2E, #2C2C2E) padding-box, linear-gradient(to right, #4fc3f7, #81c784, #ffeb3b, #ff9800, #f06292) border-box",
                        backgroundClip: "padding-box, border-box",
                      }}
                    >
                      {msg.isThinking ? (
                        <span className="inline-block animate-pulse text-2xl">
                          ...
                        </span>
                      ) : (
                        <div className="flex flex-col gap-3 whitespace-pre-wrap">
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                              p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                              strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                              ul: ({ children }) => <ul className="list-disc pl-4 space-y-1">{children}</ul>,
                              ol: ({ children }) => <ol className="list-decimal pl-4 space-y-1">{children}</ol>,
                              li: ({ children }) => <li>{children}</li>,
                              table: ({ children }) => (
                                <div className="overflow-x-auto my-4">
                                  <table className="min-w-full border-collapse border border-white/20">
                                    {children}
                                  </table>
                                </div>
                              ),
                              thead: ({ children }) => <thead className="bg-white/10">{children}</thead>,
                              tbody: ({ children }) => <tbody>{children}</tbody>,
                              tr: ({ children }) => <tr className="border-b border-white/10">{children}</tr>,
                              th: ({ children }) => (
                                <th className="px-4 py-2 text-left font-semibold text-white border border-white/20">
                                  {children}
                                </th>
                              ),
                              td: ({ children }) => (
                                <td className="px-4 py-2 text-white/90 border border-white/20">
                                  {children}
                                </td>
                              ),
                            }}
                          >
                            {msg.message}
                          </ReactMarkdown>

                          {msg.warnings?.length ? (
                            <div className="rounded-lg bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
                              <div className="font-semibold text-amber-100">
                                Warnings
                              </div>
                              <ul className="mt-1 list-disc pl-4 space-y-1">
                                {msg.warnings.map((warning, index) => (
                                  <li key={index}>{warning}</li>
                                ))}
                              </ul>
                            </div>
                          ) : null}

                          {msg.actionPlan ? (
                            <div className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs">
                              <div className="font-semibold uppercase tracking-wide text-white/70">
                                Action plan
                              </div>
                              {showPlanText ? (
                                <p className="mt-2 text-sm text-white/90">
                                  {msg.actionPlan.human_plan}
                                </p>
                              ) : null}
                              <div className="mt-2 grid gap-1 text-white/80">
                                <div>
                                  <span className="font-semibold">Type:</span>{" "}
                                  {msg.actionPlan.action_type}
                                </div>
                                {msg.actionPlan.risk_level ? (
                                  <div>
                                    <span className="font-semibold">Risk:</span>{" "}
                                    {msg.actionPlan.risk_level}
                                  </div>
                                ) : null}
                                {msg.actionPlan.estimated_time_seconds ? (
                                  <div>
                                    <span className="font-semibold">ETA:</span>{" "}
                                    {msg.actionPlan.estimated_time_seconds}s
                                  </div>
                                ) : null}
                                {msg.actionPlan.data_scope?.estimated_rows ? (
                                  <div>
                                    <span className="font-semibold">Rows:</span>{" "}
                                    {msg.actionPlan.data_scope.estimated_rows}
                                  </div>
                                ) : null}
                                {msg.actionPlan.data_scope?.date_range ? (
                                  <div>
                                    <span className="font-semibold">Range:</span>{" "}
                                    {msg.actionPlan.data_scope.date_range.start} to{" "}
                                    {msg.actionPlan.data_scope.date_range.end}
                                  </div>
                                ) : null}
                              </div>
                              {msg.actionPlan.plan_steps?.length ? (
                                <ul className="mt-2 list-disc pl-4 text-white/80">
                                  {msg.actionPlan.plan_steps.map((step, index) => (
                                    <li key={index}>{step}</li>
                                  ))}
                                </ul>
                              ) : null}
                              {msg.actionPlan.warnings?.length ? (
                                <div className="mt-2 text-amber-200">
                                  {msg.actionPlan.warnings.join(" ")}
                                </div>
                              ) : null}
                            </div>
                          ) : null}

                          {msg.actionResult ? (
                            <div className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs text-white/80">
                              <div className="font-semibold uppercase tracking-wide text-white/70">
                                Action result
                              </div>
                              <div className="mt-2">
                                <span className="font-semibold">Status:</span>{" "}
                                {msg.actionResult.status}
                              </div>
                              {msg.actionResult.message ? (
                                <div className="mt-1">{msg.actionResult.message}</div>
                              ) : null}
                              {msg.actionResult.error ? (
                                <div className="mt-1 text-red-200">
                                  {msg.actionResult.error}
                                </div>
                              ) : null}
                            </div>
                          ) : null}

                          {msg.downloads?.length ? (
                            <div className="flex flex-wrap gap-2">
                              {msg.downloads.map((download) => (
                                <Button
                                  key={download.url}
                                  variant="outline"
                                  size="sm"
                                  asChild
                                >
                                  <a
                                    href={download.url}
                                    target="_blank"
                                    rel="noreferrer"
                                  >
                                    Download {download.label}
                                  </a>
                                </Button>
                              ))}
                            </div>
                          ) : null}

                          {msg.suggestions?.length ? (
                            <div className="text-xs text-white/70">
                              Suggestions: {msg.suggestions.join(" | ")}
                            </div>
                          ) : null}

                          {showConfirmation ? (
                            <div className="flex flex-wrap gap-2">
                              <Button
                                onClick={() => handleConfirmAction(true)}
                                disabled={isConfirming}
                                className="bg-emerald-500 hover:bg-emerald-600"
                              >
                                Approve
                              </Button>
                              <Button
                                onClick={() => handleConfirmAction(false)}
                                disabled={isConfirming}
                                variant="secondary"
                                className="bg-red-500/90 hover:bg-red-600 text-white"
                              >
                                Reject
                              </Button>
                            </div>
                          ) : null}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="max-w-[60%] flex flex-col text-white bg-[#2196f3] ml-auto items-start gap-2 rounded-[20px] p-4 text-left text-base font-medium transition-all whitespace-pre-wrap">
                      {msg.message}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
          <div ref={messagesEndRef} className="mb-2"></div>
        </ScrollArea>
        <div className="w-full sm:max-w-3xl mx-auto">
          <div className="p-8">
            <div
              className="flex flex-row items-center gap-4 border-none px-4 py-3"
              style={{
                border: "3px solid transparent",
                borderRadius: "40px",
                background:
                  "linear-gradient(#262628, #262628) padding-box, linear-gradient(to right, #4fc3f7, #81c784, #ffeb3b, #ff9800, #f06292) border-box",
                backgroundClip: "padding-box, border-box",
              }}
            >
              <AutosizeTextarea
                className="flex-1 outline-none border-0 bg-transparent text-white placeholder-gray-400 text-2xl px-0"
                placeholder={inputPlaceholder}
                minHeight={25}
                maxHeight={55}
                rows={1}
                onKeyDown={(e) => handleEnter(e)}
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
                disabled={isInputDisabled}
              />
              <Button
                onClick={() => sendMessage()}
                className="h-12 w-12 p-0 bg-[#2196f3] hover:bg-blue-600 rounded-full flex items-center justify-center"
                style={{ minWidth: 48, minHeight: 48 }}
                aria-label="Send message"
                disabled={!canSend}
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="h-6 w-6 text-white"
                >
                  <line x1="22" y1="2" x2="11" y2="13" />
                  <polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
              </Button>
            </div>
          </div>
        </div>

        {/* Mobile footer branding - visible on mobile only */}
        <div className="md:hidden flex items-center justify-center gap-2 py-2 text-white/20 text-xs">
          <img src="/bunq-logo.svg" alt="bunq" className="h-4 opacity-40" />
          <span className="font-light">Developed by Emin Durmuş</span>
        </div>
      </div>
    </main>
  );
}
