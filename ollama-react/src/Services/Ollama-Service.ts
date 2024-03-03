import { Observable, Subject } from "rxjs";

// {"model":"gemma:2b","created_at":"2024-03-03T03:45:39.829495263Z","response":"2","done":false}
export type IOllamaResponse = {
    model: string,
    created_at: Date,
    response?: string,
    done: boolean,
    total_duration: number,
    load_duration: number,
    prompt_eval_count: number,
    prompt_eval_duration: number,
    eval_count: number,
    eval_duration: number
}

export type IOllamaStatistics = {
    total_duration: number,
    load_duration: number,
    prompt_eval_count: number,
    prompt_eval_duration: number,
    eval_count: number,
    eval_duration: number,
    token_per_second: number;
}

export class OllamaService {
    private newMessage$: Subject<string> = new Subject();
    private newStatistics$: Subject<IOllamaStatistics> = new Subject<IOllamaStatistics>();
    private static _instance: OllamaService;

    private constructor() {}

    public static get Instance(): OllamaService {
        if (!this._instance) {
            this._instance = new OllamaService();
        }

        return this._instance;
    }

    public async sendQuery(query: string) {
        try {
            const prompt = { role: 'User', content: query };
            const bodyRequest = { model: 'gemma:2b', prompt: JSON.stringify(prompt), streaming: true };
            const response = await fetch('/api/generate', {
                method: 'POST',
                body: JSON.stringify(bodyRequest),
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const stream = response.body as ReadableStream;
            const reader = stream.getReader();

            // Read chunks of data as they arrive
            while (true) {
                const { done, value } = await reader.read();

                if (done) {
                    console.log('Streaming ended.');
                    break;
                }

                // Process the received data (value)
                const text = new TextDecoder('utf-8').decode(value); // convert hex-bytes to utf-8
                const jsonObj: IOllamaResponse = JSON.parse(text) as IOllamaResponse;
                // expected format. {"model":"gemma:2b","created_at":"2024-03-03T03:45:39.829495263Z","response":"2","done":false}
                if (jsonObj.done) {
                    // bundle statistics
                    const statistics: IOllamaStatistics = {
                        total_duration: jsonObj.total_duration / 1E9,
                        load_duration: jsonObj.load_duration / 1E9,
                        prompt_eval_count: jsonObj.prompt_eval_count,
                        prompt_eval_duration: jsonObj.prompt_eval_duration / 1E9,
                        eval_count: jsonObj.eval_count,
                        eval_duration: jsonObj.eval_duration / 1E9,
                        token_per_second: 0
                    }
                    statistics.token_per_second = statistics.eval_count / statistics.eval_duration;
                    this.newStatistics$.next(statistics);
                }

                if (jsonObj.response) {
                    this.newMessage$.next(jsonObj.response);
                }
            }
        }
        catch (err: any) {
            console.error('Error:', err.message);
        }
    }

    public get onNewStats(): Observable<IOllamaStatistics> {
        return this.newStatistics$.asObservable();
    }

    public get onNewMessage(): Observable<string> {
        return this.newMessage$.asObservable();
    }
}